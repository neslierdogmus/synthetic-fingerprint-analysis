#!/usr/bin/env python3

from pathlib import Path

from matplotlib import image

class FCGroundTruth:
    def __init__(self, gender: str, fp_class: str, fp_class_2: str):
        """
        Initialize a FCGroundTruth object.

        Parameters:
        gender (str): The gender of the fingerprint.
        fp_class (str): The class of the fingerprint.
        fp_class_2 (str): The second class of the fingerprint.

        """
        self.gender = gender
        self.fp_class = fp_class
        self.fp_class_2 = fp_class_2

    @classmethod
    def from_file(cls, gt_path: Path):
        """
        Create a ground truth object from a file.

        Parameters:
        gt_path (Path): The path to the ground truth file.

        Returns:
        FCGroundTruth: The ground truth object read from the file.
        """
        with open(gt_path) as fin:
            try:
                gender = fin.readline().split()[1]
            except IndexError:
                gender = ''
            fp_class = fin.readline().split()[1]
            try:
                fp_class_2 = fin.readline().split()[2][1]
            except IndexError:
                fp_class_2 = ''

        return cls(gender, fp_class, fp_class_2)


class FCFingerprint:
    def __init__(self, fp_path: Path, fp_fname: str):
        """
        Initialize FCFingerprint object from a given path and filename.

        Parameters
        ----------
        fp_path : Path
            Path to the directory containing the fingerprint file.
        fp_fname : str
            Filename of the fingerprint file (without extension).

        Attributes
        ----------
        fp_path : Path
            Path to the directory containing the fingerprint file.
        fp_fname : str
            Filename of the fingerprint file (without extension).
        image : ndarray
            2D array containing the fingerprint image data.
        gt : FCGroundTruth
            FCGroundTruth object containing the ground truth data.
        """
        self.fp_path = fp_path
        self.fp_fname = fp_fname

        img_path = fp_path.joinpath(fp_fname + '.png')
        self.image = image.imread(img_path)

        gt_path = fp_path.joinpath(fp_fname + '.txt')
        self.gt = FCGroundTruth.from_file(gt_path)

    def __repr__(self):
        """
        Return a string representation of the FCFingerprint object.

        Returns
        -------
        str
            String representation of the FCFingerprint object.
        """
        return 'FCFingerprint({}, {})'.format(self.fp_path, self.fp_fname)

    def __str__(self):
        """
        Return a string representation of the FCFingerprint object.

        The string representation includes the width and height of the fingerprint
        image, the class of the fingerprint, and the filename of the fingerprint.

        Returns
        -------
        str
            String representation of the FCFingerprint object.
        """
        h, w = self.image.shape
        return '{}x{} {} FCFingerprint filename {}'.format(w, h,
                                                              self.gt.fp_class,
                                                              self.fp_fname)


if __name__ == '__main__':
    import sys
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Fingerprint loading tests')
    parser.add_argument('-p', '--fp-path', dest='fp_path',
                        default='/home/finperprint-analysis/datasets/fc/png_txt/figs_0',
                        metavar='BASEPATH',
                        help='directory for fingerprint files')
    parser.add_argument('-fn', '--fp-fname', dest='fp_fname',
                        default='f0001_01', metavar='FINGERPRINT FILE NAME',
                        help='name of the fingerprint file')

    args = parser.parse_args(sys.argv[1:])

    fp_path = Path(args.fp_path)
    fp_fname = args.fp_fname

    fp = FCFingerprint(fp_path, fp_fname)

    print('Created a fingerprint with id {}'.format(fp.fp_fname))
    print(fp)

