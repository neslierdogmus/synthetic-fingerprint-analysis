#!/usr/bin/env python3
"""Estimate the implied capture DPI of each /data/synthetic source from its
own ridge spacing, since none of them (except S07) carry usable DPI
metadata and several are plausibly GAN/diffusion output with no physical
scale concept at all (see CONTEXT.md).

Method: run pyfing's ridge-frequency estimator (Snffe) with dpi=500 (a
neutral, no-rescale call -- see Snffe.run's source: dpi==dnn_input_dpi
means f=1, so the model sees the image at its own native pixel grid and
reports ridge period directly in those pixels, not in some canonical
scale). Take the median ridge period in pixels over foreground blocks,
then convert to an implied DPI using the population-average real
fingerprint ridge period (~0.463mm, the constant used in Hong/Wan/Jain-
style fingerprint enhancement literature): implied_dpi = period_px *
25.4 / 0.463.

This is a rough signal, not a calibration -- Snffe was trained on real
(~500dpi-scale) ridge statistics, so its estimate on a source whose true
scale is very different, or whose ridge patterns don't match real
biological statistics (a risk for naive image-domain GANs), should be
read as directional, not exact.
"""
import glob
import os

import cv2 as cv
import numpy as np
import pyfing

SOURCE_EXTENSIONS = {
    "S01": "*.tif", "S02": "*.jpg", "S03": "*.jpeg", "S04": "*.JPEG", "S05": "*.jpg",
    "S06": "*.jpg", "S07": "*.png", "S08": "*.png", "S09": "*.png", "S10": "*.png", "S11": "*.png",
}
REAL_RIDGE_PERIOD_MM = 0.463
N_PER_SOURCE = 15


def implied_dpi_for_image(seg, orient, freq, image):
    mask = seg.run(image)
    if mask.sum() == 0:
        return None
    orientation, _ = orient.run(image, mask, dpi=500)
    period_px = freq.run(image, mask, orientation, dpi=500)
    fg = mask > 0
    vals = period_px[fg]
    vals = vals[(vals > 1) & (vals < 40)]  # discard degenerate outliers
    if len(vals) == 0:
        return None
    median_period_px = float(np.median(vals))
    return median_period_px, median_period_px * 25.4 / REAL_RIDGE_PERIOD_MM


def main():
    seg = pyfing.Gmfs()
    orient = pyfing.Snfoe()
    freq = pyfing.Snffe()

    print(f"{'source':6s} {'n':>4s} {'median_period_px':>18s} {'implied_dpi':>12s}  {'note'}")
    for source, pattern in SOURCE_EXTENSIONS.items():
        files = sorted(glob.glob(f"/data/synthetic/{source}/{pattern}"))[:N_PER_SOURCE]
        periods, dpis = [], []
        for f in files:
            image = cv.imread(f, cv.IMREAD_GRAYSCALE)
            if image is None:
                continue
            result = implied_dpi_for_image(seg, orient, freq, image)
            if result is None:
                continue
            period_px, dpi = result
            periods.append(period_px)
            dpis.append(dpi)
        if not periods:
            print(f"{source:6s} {'0':>4s} {'--':>18s} {'--':>12s}  no usable images")
            continue
        med_period = np.median(periods)
        med_dpi = np.median(dpis)
        note = ""
        if abs(med_dpi - 500) < 50:
            note = "consistent with 500dpi"
        elif med_dpi < 300:
            note = "much LOWER than 500dpi"
        elif med_dpi > 700:
            note = "much HIGHER than 500dpi"
        else:
            note = "moderately off from 500dpi"
        print(f"{source:6s} {len(periods):4d} {med_period:18.2f} {med_dpi:12.1f}  {note}")


if __name__ == "__main__":
    main()
