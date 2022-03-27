import yaml
from torch.utils.data import DataLoader, SubsetRandomSampler
import segmentation_models_pytorch as smp
import torchvision.transforms as T
from tensorboardX import SummaryWriter
import numpy as np

from models import define_G, define_D
from datasets import COVIDxCT, SegmentationDataset
from metrics import LossMetric, ImageLogMetric, OverlayLogMetric
from transforms import RandomPixelRescale, RandomScaleJitter

MODEL_TYPES = ('seg', 'lsgan', 'wgangp')
EXPERIMENT_DIR = 'F:\\Models\\CTSegGAN'
TORCH_HOME = 'F:\\Models\\CTSegGAN\\pretrained'


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def make_weak_dataset(config, size):
    """Makes weakly-supervised segmentation dataset and loader"""
    # Augmentation settings
    augmentations = config['augmentations']
    px_scale_change = augmentations['pixel_scale']
    px_shift = augmentations['pixel_shift']
    rotation = augmentations['rotation']
    translation = [augmentations['translation'], augmentations['translation']]
    shear = [-augmentations['shear'], augmentations['shear'], -augmentations['shear'], augmentations['shear']]

    # Make transform
    train_tform = T.Compose([
        # ExteriorExclusion(),
        T.Resize([size, size]),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        RandomScaleJitter(),
        T.RandomApply([
            T.RandomAffine(
                degrees=rotation, translate=translation, shear=shear, interpolation=T.InterpolationMode.BILINEAR),
            RandomPixelRescale(px_scale_change, px_shift, output_range=(0, 1)),
        ])
    ])

    train_dataset = COVIDxCT(config['data_dir'], config['label_file'], transform=train_tform)
    classes = np.array([sample[1] for sample in train_dataset.samples])
    pos = classes > 0
    neg = classes == 0
    pos_indices = np.where(pos)[0]
    neg_indices = np.where(neg)[0]
    pos_sampler = SubsetRandomSampler(pos_indices)
    neg_sampler = SubsetRandomSampler(neg_indices)
    pos_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                            num_workers=config['num_workers'], sampler=pos_sampler, drop_last=True)
    neg_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                            num_workers=config['num_workers'], sampler=neg_sampler, drop_last=True)

    return pos_loader, neg_loader


def make_strong_dataset(config, size):
    """Makes strongly-supervised segmentation dataset and loader"""
    # Augmentation settings
    augmentations = config['augmentations']
    px_scale_change = augmentations['pixel_scale']
    px_shift = augmentations['pixel_shift']
    rotation = augmentations['rotation']
    translation = [augmentations['translation'], augmentations['translation']]
    shear = [-augmentations['shear'], augmentations['shear'], -augmentations['shear'], augmentations['shear']]

    # Image-only transform
    img_tform = T.Compose([
        # ExteriorExclusion(),
        T.ToTensor(),
        T.Resize([size, size]),
        T.RandomApply([RandomPixelRescale(px_scale_change, px_shift, output_range=(0, 1))])
    ])

    # Mask-only transform
    seg_tform = T.Compose([
        T.ToTensor(),
        T.Resize([size, size], interpolation=T.InterpolationMode.NEAREST)
    ])

    # Image and mask transform
    joint_tform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomApply([
            T.RandomAffine(
                degrees=rotation, translate=translation, shear=shear, interpolation=T.InterpolationMode.BILINEAR)
        ]),
        RandomScaleJitter()
    ])

    # Make dataset
    seg_dataset = SegmentationDataset(
        config['data_dir'], transform=img_tform, target_transform=seg_tform, joint_transform=joint_tform)

    # Reduce size if required
    if config['fraction'] < 1:
        np.random.seed(config['seed'])
        indices = np.random.choice(
            list(range(len(seg_dataset))), round(config['fraction']*len(seg_dataset)), replace=False)
        sampler = SubsetRandomSampler(indices)
        seg_loader = DataLoader(
            seg_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], sampler=sampler)
    else:
        seg_loader = DataLoader(
            seg_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=True)
    return seg_loader


def make_val_dataset(config, size):
    """Makes segmentation dataset and loader for validation"""
    img_tform = T.Compose({
        T.ToTensor(),
        T.Resize([size, size], interpolation=T.InterpolationMode.BILINEAR)
    })
    seg_tform = T.Compose({
        T.ToTensor(),
        T.Resize([size, size], interpolation=T.InterpolationMode.NEAREST)
    })
    val_dataset = SegmentationDataset(
        config['data_dir'], transform=img_tform, target_transform=seg_tform)
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=False)

    return val_loader


def make_metrics(event_dir, log_interval, model_type='seg'):
    if model_type not in MODEL_TYPES:
        raise ValueError('model_type must be one of {}'.format(MODEL_TYPES))

    writer = SummaryWriter(logdir=event_dir)

    metrics = [LossMetric('sup_seg_loss', log_interval, loss_key='sup_seg_loss')]
    if model_type != 'seg':
        metrics.extend([
            LossMetric('disc_gan_loss', log_interval, loss_key='disc_gan_loss'),
            LossMetric('gen_gan_loss', log_interval, loss_key='gen_gan_loss'),
            LossMetric('id_loss', log_interval, loss_key='id_loss'),
            LossMetric('seg_loss', log_interval, loss_key='seg_loss'),
            LossMetric('sup_seg_loss', log_interval, loss_key='sup_seg_loss'),
            OverlayLogMetric('seg_pos', 20*log_interval, image_key='image_pos', mask_key='seg_pos', max_images=4),
            OverlayLogMetric('seg_neg', 20*log_interval, image_key='image_neg', mask_key='seg_neg', max_images=4),
            ImageLogMetric('fake_neg', 20*log_interval, image_key='fake_neg', max_images=4)
        ])
        if model_type == 'wgangp':
            metrics.append(LossMetric('disc_gp', log_interval, loss_key='disc_gp'))
    return metrics, writer


def make_models(config, mode='seg'):
    # Make segmentation model
    seg_model = smp.create_model(**config['segmentation'])

    # Make GAN model
    if mode == 'gan':
        if 'generator' in config and 'discriminator' in config:
            gen_model = define_G(**config['generator'])
            disc_model = define_D(**config['discriminator'])
            return seg_model, gen_model, disc_model
        else:
            raise ValueError('generator and discriminator must be defined for adversarial training')
    return seg_model
