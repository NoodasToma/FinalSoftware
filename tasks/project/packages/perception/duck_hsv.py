"""Dependency-free duckie detector: HSV colour + shape.

WHY this exists: the trained YOLO model can't run on the real bot — its Jetson
has Python 3.6 + OpenCV 4.1.1 (whose cv2.dnn can't parse the YOLOv5 ONNX) and no
onnxruntime. This detector needs only OpenCV core (inRange / findContours /
morphology), which works on 4.1.1, so the bot can brake for duckies with no extra
install. It returns the SAME detection format as ObjectDetectionAgent —
[((x1,y1,x2,y2), score, 0)] in full-frame pixels — so it drops straight into the
agent's existing obstacle / SOFT_STOP path (should_stop_for_obstacle, debounce).

The hard part is that the road's yellow centre line is the SAME colour as a duck.
We reject it by SHAPE, mirroring the reference's tuned approach: lane markings are
long, thin, high-extent rectangles; a duck is compact. The filters below encode
that. Tune the HSV range + thresholds on the bot via config if needed.
"""
import cv2
import numpy as np

# Defaults are starting points; override any key via the `cfg` dict (the agent
# passes config/duck_hsv from the timings, so you can tune on the bot live).
_DEFAULT = {
    'yellow_lower': [18, 90, 110],   # HSV low  (duck yellow/orange, fairly saturated+bright)
    'yellow_upper': [40, 255, 255],  # HSV high
    'top_ignore_frac': 0.15,         # blank out the top of the frame (sky/horizon)
    'min_contour_area': 200,         # px, drop specks
    'min_bbox_area_frac': 0.0015,    # px area / (w*h); drop far-tiny blobs
    'min_height_frac': 0.12,         # box height / frame height. THE main duck-vs-dash gate:
                                     # a duck close enough to brake for is tall (>~12% of the
                                     # frame); painted lane dashes are short (measured 4-10%).
    'min_y2_ratio': 0.28,            # box bottom must be below this fraction of height
    'cx_margin_frac': 0.15,          # ignore blobs hugging the left/right edge (lane lines sit
                                     # near the edges; an in-path obstacle is more central)
    'min_aspect': 0.30,              # w/h bounds. A standing rubber duck is TALLER than wide
    'max_aspect': 1.30,              # (measured ~0.68); painted dashes/markings on the ground
                                     # are WIDER than tall (~1.5-1.7). So cap aspect near 1 — the
                                     # single best duck-vs-dash discriminator on the real road.
    'max_rotated_aspect': 2.70,      # min-area-rect elongation; lane dashes are very elongated
    'max_solidity': 0.82,            # contour_area / ROTATED-rect area. A painted lane dash
                                     # fills its (possibly slanted) rect (~0.9-1.0); a rounded
                                     # rubber duck does not (~0.6-0.8). Rotation-invariant, so it
                                     # rejects perspective-slanted dashes too. Key discriminator.
    # Bottom-square reject — THE gate for venues whose centre dashes are CHUNKY/
    # SQUARE (aspect ~1.0) instead of thin: the nearest such dash fills the bottom
    # of the frame as an ~square blob touching the bottom edge (measured FP:
    # [143,393,228,480], aspect 0.977, y2=480). A standing rubber duck is taller-
    # than-wide (aspect ~0.68-0.85), so rejecting blobs that TOUCH the bottom edge
    # AND are squarish drops the dash while keeping a real close duck.
    'reject_bottom_square': True,
    'bottom_touch_frac': 0.985,      # y2 >= this*h => the blob runs off the bottom edge (a road dash)
    'bottom_square_aspect': 0.88,    # ... and aspect >= this (squarish). real duck 0.84 < 0.88 survives.
    'score_area_norm': 8000.0,       # bbox area mapped to score 1.0 (proximity proxy)
}


def detect_duckies_hsv(bgr, cfg=None, lane_xs=None):
    """bgr: HxWx3 BGR frame. Returns [((x1,y1,x2,y2), score, 0), ...].

    lane_xs: optional list of full-frame x positions of the detected YELLOW lane
    (centre) line, from the lane follower. The yellow dashed centre line is the
    SAME colour as a duck, and in some venues the dashes are chunky/square so the
    aspect/solidity gates can't reject them. But a dash sits ON the lane line and
    a real duck obstacle sits OFF it, so a yellow blob whose centre-x lands within
    `duck_lane_exclude_px` of the lane line is treated as the lane and skipped.
    Pass None (or leave the list empty) to disable this gate."""
    if bgr is None or bgr.size == 0:
        return []
    c = dict(_DEFAULT)
    if cfg:
        c.update(cfg)
    lane_excl = float(c.get('duck_lane_exclude_px', 0))
    lane_xs = [lx for lx in (lane_xs or []) if lx is not None]
    h, w = bgr.shape[:2]

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(c['yellow_lower'], np.uint8),
                            np.array(c['yellow_upper'], np.uint8))
    top = int(h * c['top_ignore_frac'])
    if top > 0:
        mask[:top, :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    found = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]   # cv2 3.x vs 4.x return arity

    out = []
    for cnt in contours:
        carea = float(cv2.contourArea(cnt))
        if carea < c['min_contour_area']:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw <= 2 or bh <= 2:
            continue
        x1, y1, x2, y2 = x, y, x + bw, y + bh
        barea = float(bw * bh)
        extent = carea / (barea + 1e-6)
        aspect = bw / float(bh + 1e-6)
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        rot = (max(rw, rh) / (min(rw, rh) + 1e-6)) if (rw > 1 and rh > 1) \
            else (max(bw, bh) / (min(bw, bh) + 1e-6))
        solidity = carea / (rw * rh + 1e-6) if (rw > 1 and rh > 1) else extent
        cx = (x1 + x2) / 2.0
        y2_ratio = y2 / float(h)

        # --- shape/position rejection (mostly to drop the yellow lane line) ---
        if bh < h * c['min_height_frac']:                      # too short = a lane dash, not a duck
            continue
        if y2_ratio < c['min_y2_ratio']:                       # too far up
            continue
        if cx < w * c['cx_margin_frac'] or cx > w * (1 - c['cx_margin_frac']):
            continue                                           # hugging an edge
        if lane_excl > 0 and lane_xs and min(abs(cx - lx) for lx in lane_xs) < lane_excl:
            continue                                           # on the yellow lane line = a dash
        if barea < w * h * c['min_bbox_area_frac']:            # too small
            continue
        if aspect > c['max_aspect'] or aspect < c['min_aspect']:
            continue
        if rot > c['max_rotated_aspect']:                      # long thin dash
            continue
        if solidity > c['max_solidity']:                       # solid filled quad = painted marking
            continue
        if c.get('reject_bottom_square') and y2 >= h * c['bottom_touch_frac'] \
                and aspect >= c['bottom_square_aspect']:
            continue                                           # chunky near dash running off the bottom edge

        score = min(1.0, barea / c['score_area_norm'])
        out.append(((x1, y1, x2, y2), float(score), 0))

    return out


# Other-bot detector by colour. The other Duckiebots here are BLUE and carry no
# AprilTag plate, so the tag path can't see them. A close bot ahead is a big blue
# blob low-centre in the frame; that's cheap and reliable to find on OpenCV 4.1.1.
# Returns [((x1,y1,x2,y2), score, 1)] (class 1 = vehicle/other-bot).
_VEH_DEFAULT = {
    'blue_lower': [95, 80, 50],
    'blue_upper': [130, 255, 255],
    'top_ignore_frac': 0.18,         # ignore blue chairs/objects up near the horizon
    'min_bbox_area_frac': 0.020,     # a bot close enough to brake for fills a good chunk
    'min_cy_frac': 0.30,             # blob centre below this fraction of height (in front of us)
    'cx_margin_frac': 0.20,          # roughly ahead, not way off to the side
    'min_fill': 0.30,                # contour/bbox; reject sparse glare/specular blue
    'min_aspect': 0.40,
    'max_aspect': 3.50,
    'score_area_norm': 30000.0,
}


def detect_vehicles_hsv(bgr, cfg=None):
    """bgr: HxWx3 BGR frame. Returns [((x1,y1,x2,y2), score, 1), ...] for blue bots."""
    if bgr is None or bgr.size == 0:
        return []
    c = dict(_VEH_DEFAULT)
    if cfg:
        c.update(cfg)
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(c['blue_lower'], np.uint8),
                            np.array(c['blue_upper'], np.uint8))
    top = int(h * c['top_ignore_frac'])
    if top > 0:
        mask[:top, :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    found = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]

    out = []
    for cnt in contours:
        carea = float(cv2.contourArea(cnt))
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw <= 2 or bh <= 2:
            continue
        barea = float(bw * bh)
        if barea < w * h * c['min_bbox_area_frac']:
            continue
        cy = (y + y + bh) / 2.0
        if cy < h * c['min_cy_frac']:
            continue
        cx = x + bw / 2.0
        if cx < w * c['cx_margin_frac'] or cx > w * (1 - c['cx_margin_frac']):
            continue
        aspect = bw / float(bh + 1e-6)
        if aspect < c['min_aspect'] or aspect > c['max_aspect']:
            continue
        if (carea / (barea + 1e-6)) < c['min_fill']:
            continue
        out.append(((x, y, x + bw, y + bh), min(1.0, barea / c['score_area_norm']), 1))
    return out
