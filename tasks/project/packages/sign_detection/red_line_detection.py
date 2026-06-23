"""
red_line_detection.py
"""

import numpy as np
import cv2


def detect_red_line(signBehavior, frame_rgb: np.ndarray) -> bool:
    h, w = frame_rgb.shape[:2]

    strip_h = max(2, int(h * signBehavior.cfg.red_strip_frac))

    y0 = h - strip_h
    y1 = h

    l = int(w * signBehavior.cfg.red_roi_left)
    r = int(w * signBehavior.cfg.red_roi_right)

    strip = frame_rgb[y0:y1, l:r]

    if strip.size == 0:
        return False

    strip_h_actual, strip_w_actual = strip.shape[:2]

    hsv = cv2.cvtColor(strip, cv2.COLOR_RGB2HSV)

    lo1 = np.array(signBehavior.cfg.red_hsv_low1, dtype=np.uint8)
    hi1 = np.array(signBehavior.cfg.red_hsv_high1, dtype=np.uint8)
    lo2 = np.array(signBehavior.cfg.red_hsv_low2, dtype=np.uint8)
    hi2 = np.array(signBehavior.cfg.red_hsv_high2, dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lo1, hi1)
    mask2 = cv2.inRange(hsv, lo2, hi2)

    mask = mask1 | mask2

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    red_pixels = int(np.count_nonzero(mask))
    strip_area = strip_h_actual * strip_w_actual

    if strip_area <= 0:
        return False

    pixel_ratio = red_pixels / float(strip_area)

    if pixel_ratio < signBehavior.cfg.red_pixel_frac:
        return False

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    close_ratio = signBehavior.cfg.red_line_close_y2_ratio

    for cnt in contours:
        area = float(cv2.contourArea(cnt))

        if area < signBehavior.cfg.red_min_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        if bw <= 1 or bh <= 1:
            continue

        y2 = y + bh
        aspect = bw / float(bh + 1e-6)

        # Red stop line should be horizontal/wide.
        if aspect < 1.6:
            continue

        # Must be wide enough.
        if bw < strip_w_actual * signBehavior.cfg.red_min_width_frac:
            continue

        # Stop before the line, not on top of it.
        if y2 < strip_h_actual * close_ratio:
            continue

        score = area * aspect

        if best is None or score > best["score"]:
            best = {
                "area": area,
                "x": x,
                "y": y,
                "w": bw,
                "h": bh,
                "y2": y2,
                "aspect": aspect,
                "score": score,
                "pixel_ratio": pixel_ratio,
            }

    if best is None:
        return False

    print(
        "[SignBehavior] red line candidate — "
        f"area={best['area']:.0f}, "
        f"bbox=({best['x']},{best['y']},{best['w']},{best['h']}), "
        f"y2_ratio={best['y2'] / strip_h_actual:.2f}, "
        f"required={close_ratio:.2f}, "
        f"aspect={best['aspect']:.2f}, "
        f"pixel_ratio={best['pixel_ratio']:.3f}"
    )

    return True