"""FOE (orientation) ground-truth I/O and pyfing-based evaluation.

The real `FOESamples`-format .gt/.fg files do NOT match the raw-uint8
H/8 x W/8 layout LABEL_FORMAT.md describes for our own synthetic output --
they're a distinct legacy binary format. This module reads that real
format so we can score our pipeline against it; it is NOT the writer used
for synthetic annotation output (see LABEL_FORMAT.md for that format).

.gt layout (confirmed against file sizes, not just docs -- see
/workspace/CONTEXT.md "FOE ground truth format" finding):
  8-byte skip, then 6x little-endian int32: border_x, border_y (equal),
  step_x, step_y (equal), width_blocks, height_blocks. Payload is
  height_blocks x width_blocks cells, row-major, each cell TWO separate
  uint8 bytes: byte0 = orientation (angle_rad = byte/256*pi), byte1 =
  weight/confidence (unused here). An earlier attempt mis-parsed each cell
  as one int16 -- looked plausible (shape came out right) but silently
  scrambled every angle value; see CONTEXT.md for why that red herring was
  so convincing. Don't revert to that.

.fg layout: ASCII, first line "<height_blocks> <width_blocks>", then that
many space-separated 0/1 foreground-mask tokens on the same grid.
"""
import struct
import numpy as np


def load_gt(path):
    """Returns (orientation_rad, weight) arrays, each shape (height_blocks, width_blocks),
    plus (border_x, border_y, step_x, step_y)."""
    with open(path, "rb") as f:
        data = f.read()
    border_x, border_y, step_x, step_y, width_blocks, height_blocks = struct.unpack("<6i", data[8:32])
    assert border_x == border_y, (border_x, border_y)
    assert step_x == step_y, (step_x, step_y)
    payload = np.frombuffer(data, dtype=np.uint8, offset=32)
    expected = width_blocks * height_blocks * 2
    assert len(payload) == expected, (len(payload), expected, path)
    payload = payload.reshape(height_blocks, width_blocks, 2)
    orientation = payload[:, :, 0].astype(np.float64) / 256.0 * np.pi
    weight = payload[:, :, 1]
    return orientation, weight, (border_x, border_y, step_x, step_y)


def load_fg(path):
    """Returns 0/1 foreground mask, shape (height_blocks, width_blocks)."""
    with open(path) as f:
        header = f.readline().split()
    height_blocks, width_blocks = int(header[0]), int(header[1])
    mask = np.loadtxt(path, skiprows=1, dtype=np.uint8)
    return mask.reshape(height_blocks, width_blocks)


def circular_mean_block(orientation_px, cy, cx, half, h, w):
    """Circular mean of a pixel-level orientation map (radians, period pi)
    over a (2*half+1)^2 window centered at (cy, cx), clipped to bounds."""
    y0, y1 = max(0, cy - half), min(h, cy + half + 1)
    x0, x1 = max(0, cx - half), min(w, cx + half + 1)
    block = orientation_px[y0:y1, x0:x1]
    cos2, sin2 = np.cos(2 * block).mean(), np.sin(2 * block).mean()
    return np.arctan2(sin2, cos2) / 2.0 % np.pi


def evaluate_image(algorithm, image_path, gt_path, fg_path, dpi=500.0):
    """Runs `algorithm` (a pyfing OrientationEstimationAlgorithm instance,
    e.g. pyfing.Snfoe()) on one image and returns per-foreground-block
    circular angle errors in degrees (1D array)."""
    import cv2 as cv

    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    h, w = image.shape

    gt_orient, _weight, (border, _by, step, _sy) = load_gt(gt_path)
    fg = load_fg(fg_path)
    height_blocks, width_blocks = fg.shape

    fg_px = np.kron(fg, np.ones((step, step), dtype=np.uint8))
    fg_px = fg_px[:h, :w]
    if fg_px.shape != (h, w):
        pad_h, pad_w = h - fg_px.shape[0], w - fg_px.shape[1]
        fg_px = np.pad(fg_px, ((0, max(0, pad_h)), (0, max(0, pad_w))))[:h, :w]
    mask_px = (fg_px * 255).astype(np.uint8)

    orientation_px, _strength = algorithm.run(image, mask_px, dpi=int(dpi))

    half = step // 2
    errors = []
    for j in range(height_blocks):
        for i in range(width_blocks):
            if not fg[j, i]:
                continue
            cy, cx = border + j * step, border + i * step
            if cy >= h or cx >= w:
                continue
            pred = circular_mean_block(orientation_px, cy, cx, half, h, w)
            gt = gt_orient[j, i]
            diff = abs(pred - gt) % np.pi
            diff = min(diff, np.pi - diff)
            errors.append(np.degrees(diff))
    return np.array(errors)
