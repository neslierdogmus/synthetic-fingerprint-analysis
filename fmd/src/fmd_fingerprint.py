#!/usr/bin/env python3

from pathlib import Path

from matplotlib import image
import numpy as np
import PIL

class FMDGroundTruth:
    def __init__(self, width: int, height: int, quality, lst_min_type, lst_min_posX,
                 lst_min_posY, lst_min_angle, lst_min_qual):
        self.width = width
        self.height = height
        self.quality = quality

        self.lst_min_type = lst_min_type
        self.lst_min_posX = lst_min_posX
        self.lst_min_posY = lst_min_posY
        self.lst_min_angle = lst_min_angle
        self.lst_min_qual = lst_min_qual

    @classmethod
    def from_file(cls, gt_path):
        npz_file = np.load(gt_path)
        width = npz_file['size_x']
        height = npz_file['size_y']
        quality = npz_file['finger_qual']

        lst_min_type = npz_file['T']
        lst_min_posX = npz_file['X']
        lst_min_posY = npz_file['Y']
        lst_min_angle = npz_file['A']
        lst_min_qual = npz_file['Q']

        return cls(width, height, quality, lst_min_type, lst_min_posX,
                   lst_min_posY, lst_min_angle, lst_min_qual)


class FMDFingerprint:
    def __init__(self, fp_path, fp_fname):
        self.fp_path = fp_path
        self.fp_fname = fp_fname

        img_path = Path(fp_path).joinpath(fp_fname + '.tif')
        try:
            self.image = image.imread(img_path)
        except PIL.UnidentifiedImageError:
            self.image = None
            # print(img_path)

        gt_path = Path(fp_path).joinpath(fp_fname + '.npz')
        self.gt = FMDGroundTruth.from_file(gt_path)

        # TODO: Check if real or synthetic from the path
        self.fp_type = 'TODO'

    def __repr__(self):
        return 'FMDFingerprint({}, {})'.format(self.fp_path, self.fp_fname)

    def __str__(self):
        h, w = self.image.shape
        return '{}x{} {} {} FCFingerprint filename {}'.format(w, h,
                                                              self.fp_type,
                                                              self.gt.quality,
                                                              self.fp_fname)

    def get_minutiae_map(self, kernel_size=13, sigma=3):
        single_min_map_list = []
        num_min = len(self.gt.lst_min_angle)
        radius = kernel_size//2

        kernel = np.zeros((kernel_size, kernel_size))
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[i, j] = np.exp(-((i-radius)**2 +
                                        (j-radius)**2)/(2 * sigma**2))

        for m in range(num_min):
            single_min_map = np.zeros_like(np.pad(self.image, radius))
            single_min_map = single_min_map.astype(float)
            x = self.gt.lst_min_posX[m] - 1 + radius
            y = self.gt.lst_min_posY[m] - 1 + radius
            single_min_map[y-radius:y+radius+1, x-radius:x+radius+1] = kernel
            single_min_map_list.append(single_min_map)

        minutiae_map = np.max(np.array(single_min_map_list), axis=0)
        minutiae_map = minutiae_map[radius:-radius, radius:-radius]
        return minutiae_map


if __name__ == '__main__':
    import sys
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Fingerprint loading tests')
    parser.add_argument('-p', '--fp-path', dest='fp_path',
                        default='datasets/fmd/FM3_FVC2002DB1A_np',
                        metavar='BASEPATH',
                        help='directory for fingerprint files')
    parser.add_argument('-fn', '--fp-fname', dest='fp_fname',
                        default='1_1', metavar='FINGERPRINT FILE NAME',
                        help='name of the fingerprint file')

    args = parser.parse_args(sys.argv[1:])

    fp_path = args.fp_path
    fp_fname = args.fp_fname

    fp = FMDFingerprint(fp_path, fp_fname)

    minutiae_map = fp.get_minutiae_map()

    print('Created a {} fingerprint with id {}'.format(fp.fp_type,
                                                       fp.fp_fname))
    print(fp)
