#!/usr/bin/env python3

import torch
import torch.nn as nn


class FMDConvNet(nn.Module):
    def __init__(self, size=32):
        super(FMDConvNet, self).__init__()

        self.inc = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, size, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(size),
            # Defining a 2D convolution layer
            nn.Conv2d(size, size, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(size)
            )

        self.down1 = nn.Sequential(
            # 1st MaxPool -> 1/2
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining a 2D convolution layer
            nn.Conv2d(size, 2*size, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*size),
            # Defining a 2D convolution layer
            nn.Conv2d(2*size, 2*size, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*size)
            )

        self.down2 = nn.Sequential(
            # 2nd MaxPool -> 1/4
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.5),
            # Defining a 2D convolution layer
            nn.Conv2d(2*size, 4*size, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4*size),
            # Defining a 2D convolution layer
            nn.Conv2d(4*size, 4*size, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4*size)
        )

        self.uturn = nn.Sequential(
            # 3rd MaxPool -> 1/8
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.5),
            # Defining a 2D convolution layer
            nn.Conv2d(4*size, 8*size, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8*size),
            # Defining a 2D convolution layer
            nn.Conv2d(8*size, 8*size, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8*size),
            # Up-sampling -> 1/4
            nn.Dropout2d(0.5),
            nn.ConvTranspose2d(8*size, 8*size, kernel_size=2, stride=2)
        )

        self.up1 = nn.Sequential(
            # Defining a 2D convolution layer 
            nn.Conv2d((8+4)*size, 4*size, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4*size),
            # Defining a 2D convolution layer
            nn.Conv2d(4*size, 4*size, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4*size),
            # Up-sampling -> 1/2
            nn.Dropout2d(0.5),
            nn.ConvTranspose2d(4*size, 4*size, kernel_size=2, stride=2)
        )

        self.up2 = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d((4+2)*size, 2*size, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*size),
            # Defining a 2D convolution layer
            nn.Conv2d(2*size, 2*size, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(2*size),
            # Up-sampling -> 1/1
            nn.ConvTranspose2d(2*size, 2*size, kernel_size=2, stride=2)
        )

        self.outc = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d((2+1)*size, size, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(size),
            # Defining a 2D convolution layer
            nn.Conv2d(size, size, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(size),
            # Defining a 2D convolution layer
            nn.Conv2d(size, 1, kernel_size=1, stride=1, padding='same')
        )

        self.apply(self.init_weights)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.uturn(x3)
        x = self.up1(torch.cat([x, x3], dim=1))
        x = self.up2(torch.cat([x, x2], dim=1))
        x = self.outc(torch.cat([x, x1], dim=1))
        return x

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()


if __name__ == '__main__':
    import os
    import sys
    import random
    from argparse import ArgumentParser

    import numpy as np

    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    from fmd_fp_dataset import FMDFPImageDataset

    parser = ArgumentParser(description='ConvNet model tests with FOE images')
    parser.add_argument('-b', '--base-path', dest='base_path',
                        default='./datasets/fmd',
                        metavar='BASEPATH',
                        help='root directory for dataset files')
    parser.add_argument('-N', '--n-classes', dest='n_classes',
                        default=1, type=int, metavar='Nc',
                        help='number of classes')
    parser.add_argument('-nf', '--num-folds', dest='num_folds',
                        default=5, type=int, metavar='NUMFOLDS',
                        help='number of folds')
    args = parser.parse_args(sys.argv[1:])

    base_path = args.base_path
    n_classes = args.n_classes
    num_folds = args.num_folds

    fp_ids = list(range(400))
    random.shuffle(fp_ids)
    splits = np.array(np.array_split(fp_ids, num_folds))

    for fold in range(num_folds):
        fp_ids_val = splits[fold]
        fp_ids_tra = np.append(splits[:fold], splits[fold+1:])

        fmd_img_ds_val = FMDFPImageDataset(base_path, fp_ids_val, n_classes)
        fmd_img_ds_tra = FMDFPImageDataset(base_path, fp_ids_tra, n_classes)

        x, y = fmd_img_ds_val.__getitem__(0)[0:2]
        model = FMDConvNet()
        model.eval()
        y_est = model(x)

        print(x.shape)
        print(y.shape)
        print(y_est.shape)
