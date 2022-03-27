import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import functional as F
from PIL import Image


class RandomPixelRescale(nn.Module):
    def __init__(self, max_scale_change, max_shift, output_range=None):
        super().__init__()
        self.max_scale_change = min(abs(max_scale_change), 1)
        self.max_shift = abs(max_shift)
        self.output_range = output_range

    def forward(self, x):
        dtype = x.dtype
        scale = 1 + 2*self.max_scale_change*torch.rand(1, dtype=dtype) - self.max_scale_change
        shift = 2*self.max_shift*torch.rand(1, dtype=dtype) - self.max_shift

        if not torch.is_floating_point(x):
            x = x.astype(torch.float32)
        x = x*scale + shift

        if self.output_range is not None:
            x = torch.clamp(x, self.output_range[0], self.output_range[1])

        if x.dtype != dtype:
            x = x.type(dtype)

        return x


class RandomScaleJitter(nn.Module):
    """Implements random scale jitter"""
    def __init__(self, scale_range=(0.8, 1.2), p=0.5, fill=0):
        super().__init__()
        if scale_range[0] > scale_range[1]:
            raise ValueError('Invalid scale range: {}'.format(scale_range))

        self.fill = fill
        self.scale_range = scale_range
        self.p = p

    def forward(self, image):
        if torch.rand(1) > self.p:
            return image

        # Get new image size from scale
        width, height = F.get_image_size(image)
        scale = self.scale_range[0] + torch.rand(1)*(self.scale_range[1] - self.scale_range[0])
        new_width = int(width*scale)
        new_height = int(height*scale)

        if scale < 1:
            # Resize and pad image
            right_pad = width - new_width
            bottom_pad = height - new_height
            image = F.resize(image, [new_height, new_width])
            image = F.pad(image, [0, 0, right_pad, bottom_pad], fill=self.fill)
        elif scale > 1:
            # Resize and randomly crop image
            top = int(torch.rand(1)*(new_height - height))
            left = int(torch.rand(1)*(new_width - width))
            image = F.resize(image, [new_height, new_width])
            image = F.crop(image, top, left, height, width)

        return image


def body_contour(binary_image):
    """Helper function to get body contour"""
    contours = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0]
    if len(contours):
        areas = [cv2.contourArea(cnt) for cnt in contours]
        body_idx = np.argmax(areas)
        body_cont = contours[body_idx]
    else:
        body_cont = None
    return body_cont


class ExteriorExclusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if isinstance(x, Image.Image):
            x_np = np.array(x)
            x_np = exterior_exclusion(x_np)
            x = Image.fromarray(x_np)
        elif isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
            if torch.is_floating_point(x):
                x_np = np.uint8(255*x_np)
            elif x.dtype != torch.uint8:
                raise ValueError('x must have uint8 or float dtype')
            x_np = exterior_exclusion(x_np[0, 0])[None, None]
            x = torch.as_tensor(x_np, dtype=x.dtype, device=x.device)
        return x


class RandomExteriorExclusion(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.ext_exclusion = ExteriorExclusion()

    def forward(self, x):
        if torch.rand(1) < self.p:
            x = self.ext_exclusion(x)
        return x


def exterior_exclusion(image):
    """Removes visual features exterior to the patient's body"""
    # Create initial binary image
    filt_image = cv2.GaussianBlur(image, (5, 5), 0)
    filt_image.shape = image.shape  # ensure channel dimension is preserved if present
    thresh = cv2.threshold(filt_image[filt_image > 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    bin_image = filt_image > thresh

    # Find body contour
    body_cont = body_contour(bin_image.astype(np.uint8))
    if body_cont is None:
        return image

    # Exclude external regions by replacing with bg mean
    body_mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(body_mask, [body_cont], 0, 1, -1)
    body_mask = body_mask.astype(bool)
    # bg_mask = (~body_mask) & (image > 0)
    # bg_dark = bg_mask & (~bin_image)  # exclude bright regions from mean
    # bg_mean = np.mean(image[bg_dark])
    # image[bg_mask] = bg_mean
    image[~body_mask] = 0
    return image
