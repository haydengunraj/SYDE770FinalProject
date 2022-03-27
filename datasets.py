import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from data_utils import grayscale_pil_loader, auto_body_crop


class COVIDxCT(Dataset):
    def __init__(self, root, label_file, transform=None, target_transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.samples = self._get_samples(label_file)
        if len(self.samples) == 0:
            raise RuntimeError('Found 0 image files in: ' + self.root + '\n')

    def _get_samples(self, split_file):
        """Gets image filenames, classes, and bboxes"""
        samples = []
        if split_file is not None:
            with open(split_file, 'r') as f:
                for line in f.readlines():
                    fname, cls, xmin, ymin, xmax, ymax = line.strip('\n').split()
                    samples.append((
                        os.path.join(self.root, fname),
                        int(cls),
                        [int(xmin), int(ymin), int(xmax), int(ymax)]
                    ))
        return samples

    def __getitem__(self, index):
        image_path, label, bbox = self.samples[index]
        image = grayscale_pil_loader(image_path).crop(bbox)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.samples)


class SegmentationDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None, joint_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.img_dir = os.path.join(root, 'images')
        self.mask_dir = os.path.join(root, 'masks')
        img_files = sorted(glob.glob(os.path.join(self.img_dir, '*.png')))
        mask_files = sorted(glob.glob(os.path.join(self.mask_dir, '*.png')))
        self.samples = [(f_img, f_mask) for f_img, f_mask in zip(img_files, mask_files)]

    def __getitem__(self, index):
        img_path, mask_path = self.samples[index]
        image = grayscale_pil_loader(img_path)
        mask = grayscale_pil_loader(mask_path)
        _, bbox = auto_body_crop(np.asarray(image))
        image = image.crop(bbox)
        mask = mask.crop(bbox)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        if self.joint_transform is not None:
            joined = torch.cat((image, mask), dim=0)
            joined = self.joint_transform(joined)
            image = joined[:1, ...]
            mask = joined[1:, ...]
        return image, mask

    def __len__(self):
        return len(self.samples)
