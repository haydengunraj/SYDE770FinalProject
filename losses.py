import torch
import torch.nn as nn

from models import GANLoss


class Losses(nn.Module):
    def __init__(self, gan_weight, id_weight, seg_weight, gan_mode='wgangp'):
        super().__init__()
        self.gan_weight = gan_weight
        self.id_weight = id_weight
        self.seg_weight = seg_weight
        self.gan_loss = GANLoss(gan_mode)

    def discriminator_loss(self, disc_real_neg, disc_fake_neg):
        return 0.5*self.gan_weight*(self.gan_loss(disc_fake_neg, False) + self.gan_loss(disc_real_neg, True))

    def generator_loss(self, real_pos, real_neg, seg_pos, fake_neg, same_neg, disc_fake_neg):
        # GAN loss - maximize discriminator predictions for generated images
        gen_gan_loss = self.gan_weight*self.gan_loss(disc_fake_neg, True)

        # Identity loss - do not change negative images or unsegmented regions of positive images
        id_loss = torch.mean(torch.abs(real_neg - same_neg))
        id_loss += torch.mean((1. - seg_pos)*torch.abs(real_pos - fake_neg))
        id_loss *= self.id_weight

        return gen_gan_loss, id_loss

    def segmentation_loss(self, seg_pos, seg_neg):
        # Don't segment negative images
        seg_loss = torch.mean(seg_neg)

        # Penalize positive segmentations to encourage minimal segmenting
        seg_loss += 0.5*torch.mean(seg_pos)

        seg_loss *= self.seg_weight

        return seg_loss


def soft_dice_loss(preds, targets, smooth=1):
    num = targets.size(0)
    m1 = preds.view(num, -1)
    m2 = targets.view(num, -1)
    intersection = (m1 * m2)
    score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
    score = 1 - score.sum() / num
    return score
