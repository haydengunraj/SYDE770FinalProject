import os
import argparse
import torch
from torch.optim import Adam, RMSprop
from torch.cuda.amp import GradScaler, autocast

from models import cal_gradient_penalty
from losses import Losses, soft_dice_loss
from metrics import update_and_log_metrics
from eval import evaluate_segmentation
from schedulers import ConstantCosineAnnealingLR

from utils import (make_strong_dataset, make_weak_dataset, make_val_dataset,
                   make_metrics, make_models, load_config, EXPERIMENT_DIR, TORCH_HOME)

_N_CRITIC = 5  # 10 maybe?


parser = argparse.ArgumentParser()
parser.add_argument('experiment_name', type=str, help='Name of experiment')
parser.add_argument('-dv', '--device', type=str, default='cuda:0', help='Device code')
args = parser.parse_args()

os.environ['TORCH_HOME'] = TORCH_HOME

# Load configuration
experiment_dir = os.path.join(EXPERIMENT_DIR, args.experiment_name)
config_file = os.path.join(experiment_dir, 'config.yaml')
config = load_config(config_file)

# Make models
seg_model, gen_model, disc_model = make_models(config['models'], mode='gan')
seg_model.to(args.device)
gen_model.to(args.device)
disc_model.to(args.device)
seg_model.train()
gen_model.train()
disc_model.train()

# Create experiment directories
ckpt_dir = os.path.join(experiment_dir, 'checkpoints')
event_dir = os.path.join(experiment_dir, 'events')
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(event_dir, exist_ok=True)

# Load checkpoint
if config.get('checkpoint', ''):
    print('Loading weights from', config['checkpoint'])
    weights = torch.load(config['checkpoint'])
    seg_model.load_state_dict(weights['seg_model'])
    if 'gen_model' in weights:
        gen_model.load_state_dict(weights['gen_model'])
    if 'disc_model' in weights:
        disc_model.load_state_dict(weights['disc_model'])

# Create experiment directories
ckpt_dir = os.path.join(experiment_dir, 'checkpoints')
event_dir = os.path.join(experiment_dir, 'events')
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(event_dir, exist_ok=True)

# Make metrics and writer to log and track values
metrics, writer = make_metrics(event_dir, config['log_interval'], model_type='seg')

# Make datasets
pos_loader, neg_loader = make_weak_dataset(config['weak_dataset'], config['size'])
val_loader = make_val_dataset(config['val_dataset'], config['size'])
supervised = config['strong_dataset']['fraction'] > 0
if supervised:
    sup_loader = make_strong_dataset(config['strong_dataset'], config['size'])
    sup_iter = iter(sup_loader)

# Make optimizers
optimizer_g = Adam(list(gen_model.parameters()) + list(seg_model.parameters()),
                   lr=config['learning_rate'], betas=(0, 0.999))
optimizer_d = Adam(disc_model.parameters(), lr=config['learning_rate'], betas=(0, 0.999))
# optimizer_g = RMSprop(
#     list(gen_model.parameters()) + list(seg_model.parameters()), lr=args.learning_rate)
# optimizer_d = RMSprop(disc_model.parameters(), lr=args.learning_rate)
scaler_gen = GradScaler()
scaler_disc = GradScaler()

# Make loss functions
losses = Losses(config['gan_weight'], config['id_weight'], config['seg_weight'], gan_mode=config['gan_mode'])
losses.to(args.device)

# Make scheduler
gen_scheduler = ConstantCosineAnnealingLR(
    optimizer_g, config['epochs'], config['learning_rate'], alpha=config['alpha'], const_frac=config['const_lr_frac'])
disc_scheduler = ConstantCosineAnnealingLR(
    optimizer_d, config['epochs'], config['learning_rate'], alpha=config['alpha'], const_frac=config['const_lr_frac'])

step = 0
for epoch in range(config['epochs']):
    seg_model.train()

    # Log LR
    writer.add_scalar('train/LR', gen_scheduler.current_lr, step)

    # Train for an epoch
    print('\nStarting epoch {}'.format(epoch + 1))
    for real_pos, real_neg in zip(pos_loader, neg_loader):
        real_pos = real_pos[0].to(args.device)
        real_neg = real_neg[0].to(args.device)

        if supervised:
            try:
                seg_img, seg_gt = next(sup_iter)
            except StopIteration:
                sup_iter = iter(sup_loader)
                seg_img, seg_gt = next(sup_iter)
            seg_img = seg_img.to(args.device)
            seg_gt = seg_gt.to(args.device)

        ### GENERATOR TRAINING ###
        optimizer_g.zero_grad()

        with autocast():
            # Make fake negative images
            seg_pos = seg_model(real_pos)
            fake_neg_orig = gen_model(torch.cat([real_pos, seg_pos], dim=1))
            # fake_neg = fake_neg_orig
            fake_neg = seg_pos*fake_neg_orig + (1. - seg_pos)*real_pos

            # Make same negative images
            seg_neg = seg_model(real_neg)
            same_neg = gen_model(torch.cat([real_neg, torch.zeros_like(real_neg)], dim=1))

            # Run disriminator on generated images
            disc_fake_neg = disc_model(fake_neg)

            # Get generator loss
            gen_gan_loss, id_loss = losses.generator_loss(
                real_pos, real_neg, seg_pos.detach(), fake_neg, same_neg, disc_fake_neg)

            # Get segmentation loss
            seg_loss = losses.segmentation_loss(seg_pos, seg_neg)

            # Update data dict
            data_dict = {
                'image_pos': real_pos,
                'image_neg': real_neg,
                'seg_pos': seg_pos,
                'seg_neg': seg_neg,
                'fake_neg': fake_neg,
                'gen_gan_loss': gen_gan_loss,
                'id_loss': id_loss,
                'seg_loss': seg_loss
            }

            # Get supervised segmentation loss
            gen_loss = gen_gan_loss + id_loss + seg_loss
            if supervised:
                seg_pred = seg_model(seg_img)
                sup_seg_loss = soft_dice_loss(seg_pred, seg_gt)
                gen_loss += sup_seg_loss
                data_dict['sup_seg_loss'] = sup_seg_loss

        # Backprop losses and update
        if not (step + 1) % _N_CRITIC:
            scaler_gen.scale(gen_loss).backward()
            scaler_gen.step(optimizer_g)
            scaler_gen.update()

        ### DISCRIMINATOR TRAINING ###
        optimizer_d.zero_grad()

        with autocast():
            # Discriminate fake negative images
            fake_neg_detached = fake_neg.detach()
            disc_fake_neg = disc_model(fake_neg_detached)

            # Discriminate real negative images
            disc_real_neg = disc_model(real_neg)

            # Get discriminator loss
            # ## TEST ##
            # disc_real_pos = disc_model(real_pos)
            # ##########

            disc_gan_loss = losses.discriminator_loss(disc_real_neg, disc_fake_neg)

            # Get gradient penalty
            disc_loss = disc_gan_loss
            data_dict['disc_gan_loss'] = disc_gan_loss
            if config['gan_mode'] == 'wgangp':
                grad_penalty, _ = cal_gradient_penalty(
                    disc_model, real_neg, fake_neg_detached, args.device, type='mixed', constant=1.0, lambda_gp=10.0)
                disc_loss += grad_penalty
                data_dict['disc_gp'] = grad_penalty

        # Backprop loss and update
        scaler_disc.scale(disc_loss).backward()
        scaler_disc.step(optimizer_d)
        scaler_disc.update()

        # Clip weights
        # for p in disc_model.parameters():
        #     p.data.clamp_(-_GRAD_CLIP, _GRAD_CLIP)

        # Update metrics and log
        step += 1
        update_and_log_metrics(epoch + 1, step, data_dict, metrics, writer)

    # Update LR
    gen_scheduler.step()
    disc_scheduler.step()

    # Save checkpoint
    if not (epoch + 1) % config['save_interval']:
        ckpt_data = {
            'seg_model': seg_model.state_dict(),
            'gen_model': gen_model.state_dict(),
            'disc_model': disc_model.state_dict()
        }
        torch.save(ckpt_data, os.path.join(ckpt_dir, 'checkpoint_epoch-{:04d}.pth'.format(epoch + 1)))

        evaluate_segmentation(seg_model, val_loader, writer=writer, step=step, num_vis=4, device=args.device)
