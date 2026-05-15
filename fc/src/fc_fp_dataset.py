#!/usr/bin/env python3

from pathlib import Path
import random
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from fc_fingerprint import FCFingerprint


FC_CLASS_LABELS = ('T', 'A', 'L', 'R', 'W')
FC_HORIZONTAL_FLIP_LABEL_MAP = {
    'L': 'R',
    'R': 'L',
}


def encode_fc_class_names(class_names, n_classes=5):
    encoded = np.zeros(n_classes, dtype=float)
    for class_name in class_names:
        if class_name in FC_CLASS_LABELS:
            encoded[FC_CLASS_LABELS.index(class_name)] = 1.0
    return torch.from_numpy(encoded.astype(float))


def remap_fc_class_name_for_horizontal_flip(class_name):
    return FC_HORIZONTAL_FLIP_LABEL_MAP.get(class_name, class_name)


class FCFPDataset(Dataset):
    def __init__(self, base_path, fp_ids, n_classes=5):
        super(FCFPDataset).__init__()

        self.base_path = base_path
        self.fp_ids = fp_ids
        self.n_classes = n_classes
        self._hflip = False
        self._rotation_degrees = 0
        self._resize = False
        self._brightness = 0.0
        self._contrast = 0.0
        self._gaussian_blur_probability = 0.0
        self._gaussian_blur_kernel_size = 3
        self._gaussian_blur_sigma = (0.1, 1.0)
        self.fps = []

        for fp_id in fp_ids:
            fp_folder = 'figs_'+str(fp_id // 250)
            fp_path = Path(base_path).joinpath('png_txt', fp_folder)
            fp_fid = str(fp_id + 1).zfill(4)
            for fp_img_path in fp_path.glob('*'+fp_fid+'*.png'):
                fp_fname = fp_img_path.parts[-1].split('.')[0]
                fp = FCFingerprint(fp_path, fp_fname)
                self.fps.append(fp)

    def __getitem__(self, index):
        fc_fingerprint = self.fps[index]
        x = fc_fingerprint.image
        gender = fc_fingerprint.gt.gender
        fp_class = fc_fingerprint.gt.fp_class
        fp_class_2 = fc_fingerprint.gt.fp_class_2
        class_names = [fp_class, fp_class_2]

        x = torch.from_numpy(x.astype(np.single))
        x = torch.unsqueeze(x, 0)

        if self._resize:
            x = T.functional.resize(x, [224, 224], antialias=True)

        if self._hflip and random.random() >= 0.5:
            x = T.functional.hflip(x)
            class_names = [remap_fc_class_name_for_horizontal_flip(class_name)
                           for class_name in class_names]

        if self._rotation_degrees > 0:
            angle = random.uniform(-self._rotation_degrees,
                                   self._rotation_degrees)
            x = T.functional.rotate(x, angle,
                                    interpolation=T.InterpolationMode.BILINEAR)

        if self._brightness > 0:
            brightness_factor = random.uniform(1 - self._brightness,
                                               1 + self._brightness)
            x = T.functional.adjust_brightness(x, brightness_factor)

        if self._contrast > 0:
            contrast_factor = random.uniform(1 - self._contrast,
                                             1 + self._contrast)
            x = T.functional.adjust_contrast(x, contrast_factor)

        if (self._gaussian_blur_probability > 0 and
                random.random() < self._gaussian_blur_probability):
            sigma_min, sigma_max = self._gaussian_blur_sigma
            sigma = random.uniform(sigma_min, sigma_max)
            x = T.functional.gaussian_blur(
                x,
                kernel_size=self._gaussian_blur_kernel_size,
                sigma=sigma,
            )

        y = encode_fc_class_names(class_names, self.n_classes)

        return x, y, class_names[1], gender, index

    def __len__(self):
        return len(self.fps)

    def set_hflip(self, value=True):
        self._hflip = value

    def set_rotate(self, max_degrees=20):
        self._rotation_degrees = max(0, max_degrees)

    def set_color_jitter(self, brightness=0.0, contrast=0.0):
        self._brightness = max(0.0, float(brightness))
        self._contrast = max(0.0, float(contrast))

    def set_gaussian_blur(self, kernel_size=3, sigma=(0.1, 1.0), probability=0.0):
        self._gaussian_blur_kernel_size = int(kernel_size)
        self._gaussian_blur_sigma = tuple(float(value) for value in sigma)
        self._gaussian_blur_probability = max(0.0, float(probability))

    def set_resize(self, resize=True):
        self._resize = resize


if __name__ == '__main__':
    import sys
    import os
    from argparse import ArgumentParser

    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

    parser = ArgumentParser(description='FC fingerprint image dataset tests')
    parser.add_argument('-b', '--base-path', dest='base_path',
                        default='datasets/fc',
                        metavar='BASEPATH',
                        help='root directory for dataset files')
    parser.add_argument('-nf', '--num-folds', dest='num_folds',
                        default=5, type=int, metavar='NUMFOLDS',
                        help='number of folds')
    args = parser.parse_args(sys.argv[1:])

    base_path = args.base_path
    num_folds = args.num_folds

    fp_ids = list(range(500))
    random.shuffle(fp_ids)
    splits = np.array(np.array_split(fp_ids, num_folds))

    for fold in range(num_folds):
        fp_ids_val = splits[fold]
        fp_ids_tra = np.append(splits[:fold], splits[fold+1:])

        fc_fp_ds_val = FCFPDataset(base_path, fp_ids_val)
        fc_fp_ds_tra = FCFPDataset(base_path, fp_ids_tra)

        print('Created FC dataset for validation with {} fingerprints'
              .format(len(fc_fp_ds_val)))
        print(fc_fp_ds_val.fp_ids)

        print('Created FC dataset for training with {} fingerprints'
              .format(len(fc_fp_ds_tra)))
        print(fc_fp_ds_tra.fp_ids)

        print('-' * 80)
