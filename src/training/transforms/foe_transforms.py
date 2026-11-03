from __future__ import annotations

import random

import cv2
import numpy as np


class FOESpatialTransform:
    """
    Spatial augmentations for FOE images + orientation maps.

    Applies horizontal flip, rotation, centre crop, and final resize while
    keeping the orientation map values consistent (θ → π-θ on flip,
    θ → (θ+α) mod π on rotation).

    The orientation map operates at 1/8 of the image resolution.
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        hflip_p: float = 0.5,
        rotate_limit: int = 20,
        rotate_p: float = 0.5,
        crop_p: float = 0.3,
    ):
        self.input_size = input_size
        self.hflip_p = hflip_p
        self.rotate_limit = rotate_limit
        self.rotate_p = rotate_p
        self.crop_p = crop_p

    def __call__(
        self,
        image: np.ndarray,        # H × W  uint8 or float
        orientation: np.ndarray,  # H_o × W_o  float32, values in [0, π)
        mask: np.ndarray,         # H_o × W_o  float32
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (image, orientation, mask) after augmentation."""

        # ── 1. Horizontal flip ───────────────────────────────────────────────
        if random.random() < self.hflip_p:
            image = np.fliplr(image)
            orientation = np.fliplr(orientation)
            mask = np.fliplr(mask)
            nonzero = orientation > 0
            orientation[nonzero] = np.pi - orientation[nonzero]

        # ── 2. Rotation ──────────────────────────────────────────────────────
        if random.random() < self.rotate_p:
            angle_deg = random.uniform(-self.rotate_limit, self.rotate_limit)
            angle_rad = np.deg2rad(angle_deg)

            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
            image = cv2.warpAffine(
                image, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            h_o, w_o = orientation.shape
            M_o = cv2.getRotationMatrix2D((w_o / 2, h_o / 2), angle_deg, 1.0)
            orientation = cv2.warpAffine(orientation, M_o, (w_o, h_o),
                                         flags=cv2.INTER_LINEAR)
            mask = cv2.warpAffine(mask, M_o, (w_o, h_o),
                                  flags=cv2.INTER_NEAREST)

            # Update orientation values and zero background
            orientation = (orientation + angle_rad) % np.pi
            orientation *= mask

        # ── 3. Centre crop ───────────────────────────────────────────────────
        if random.random() < self.crop_p:
            H_target, W_target = self.input_size
            h, w = image.shape[:2]
            y1 = max(0, (h - H_target) // 2)
            x1 = max(0, (w - W_target) // 2)
            image = image[y1: y1 + H_target, x1: x1 + W_target]

            H_o_target = H_target // 8
            W_o_target = W_target // 8
            h_o, w_o = orientation.shape
            y1_o = max(0, (h_o - H_o_target) // 2)
            x1_o = max(0, (w_o - W_o_target) // 2)
            orientation = orientation[y1_o: y1_o + H_o_target,
                                      x1_o: x1_o + W_o_target]
            mask = mask[y1_o: y1_o + H_o_target, x1_o: x1_o + W_o_target]

        # ── 4. Resize image to target (orientation map at native res) ────────
        H_target, W_target = self.input_size
        if image.shape[:2] != (H_target, W_target):
            image = cv2.resize(image, (W_target, H_target),
                               interpolation=cv2.INTER_LINEAR)

        return image, orientation, mask
