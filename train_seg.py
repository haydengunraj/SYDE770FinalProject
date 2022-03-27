import os
import argparse
import torch
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast

from losses import soft_dice_loss
from metrics import update_and_log_metrics
from eval import evaluate_segmentation
from schedulers import ConstantCosineAnnealingLR

from utils import (make_strong_dataset, make_val_dataset, make_metrics,
                   make_models, load_config, EXPERIMENT_DIR, TORCH_HOME)


parser = argparse.ArgumentParser()
parser.add_argument('experiment_name', type=str, help='Name of experiment')
parser.add_argument('-dv', '--device', type=str, default='cuda:0', help='Device code')
args = parser.parse_args()

os.environ['TORCH_HOME'] = TORCH_HOME

# Load configuration
experiment_dir = os.path.join(EXPERIMENT_DIR, args.experiment_name)
config_file = os.path.join(experiment_dir, 'config.yaml')
config = load_config(config_file)

# Make model
seg_model = make_models(config['models'], mode='seg')
seg_model.to(args.device)
seg_model.train()

# Create experiment directories
ckpt_dir = os.path.join(experiment_dir, 'checkpoints')
event_dir = os.path.join(experiment_dir, 'events')
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(event_dir, exist_ok=True)

# Make metrics and writer to log and track values
metrics, writer = make_metrics(event_dir, config['log_interval'], model_type='seg')

# Make datasets
val_loader = make_val_dataset(config['val_dataset'], config['size'])
sup_loader = make_strong_dataset(config['strong_dataset'], config['size'])

# Make optimizer
optimizer = Adam(seg_model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.999))
scaler = GradScaler()

# Adjust epochs for roughly equal training lengths
epochs = round(config['epochs']/config['strong_dataset']['fraction'])
save_interval = round(config['save_interval']/config['strong_dataset']['fraction'])

# Make scheduler
scheduler = ConstantCosineAnnealingLR(
    optimizer, epochs, config['learning_rate'], alpha=config['alpha'], const_frac=config['const_lr_frac'])

step = 0
for epoch in range(epochs):
    seg_model.train()

    # Log LR
    writer.add_scalar('train/LR', scheduler.current_lr, step)

    # Train for an epoch
    print('\nStarting epoch {}'.format(epoch + 1))
    for seg_img, seg_gt in sup_loader:
        seg_img = seg_img.to(args.device)
        seg_gt = seg_gt.to(args.device)
        optimizer.zero_grad()

        with autocast():
            # Get supervised segmentation loss
            seg_pred = seg_model(seg_img)
            loss = soft_dice_loss(seg_pred, seg_gt)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update data dict
        data_dict = {'sup_seg_loss': loss}

        # Update metrics and log
        step += 1
        update_and_log_metrics(epoch + 1, step, data_dict, metrics, writer)

    # Update LR
    scheduler.step()

    # Save checkpoint
    if not (epoch + 1) % save_interval:
        ckpt_data = seg_model.state_dict()
        torch.save(ckpt_data, os.path.join(ckpt_dir, 'checkpoint_epoch-{:04d}.pth'.format(epoch + 1)))

        evaluate_segmentation(seg_model, val_loader, writer=writer, step=step, num_vis=4, device=args.device)
