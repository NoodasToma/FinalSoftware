from typing import Tuple

# Path to the trained model weights (.onnx file).
# Relative paths resolve from the project root.
MODEL_PATH = "tasks/object_detection/models/best.onnx"


def NUMBER_FRAMES_SKIPPED() -> int:
    # 0 = process EVERY frame handed to detect(). The project agent now runs
    # detection in a decoupled background thread fed by a maxsize-1 queue, so the
    # enqueue cadence (and the queue dropping stale frames) already throttles how
    # often inference runs. A second skip here would just drop frames the worker
    # already pulled, adding latency before a duckie is seen. Throttle via the
    # agent's detect_every / the queue, not here.
    return 0


def filter_by_classes(pred_class: int) -> bool:
    """Return False to drop this prediction."""
    # Only duckies (class 0) are pedestrians the bot must avoid; trucks and
    # signs are off the road, so drop them. (Other Duckiebots are handled by the
    # agent via their Vehicle AprilTag, not this model.)
    return pred_class == 0


def filter_by_scores(score: float) -> bool:
    """Confidence in [0.0, 1.0]. Return False to drop low-confidence boxes."""
    # 0.5 (was 0.6): a duckie is a pedestrian — better to over-stop on a slightly
    # uncertain box than to drive through one. Matches the reference's 0.45-0.5.
    return score >= 0.5


def filter_by_bboxes(bbox: Tuple[int, int, int, int]) -> bool:
    """bbox is (xmin, ymin, xmax, ymax) in pixels. Return False to drop."""
    # Drop tiny boxes (far-off duckies not in the bot's path). 600 px (was 800)
    # so a duckie is picked up a touch earlier; the stop logic (y2/area gate in
    # obstacles.should_stop_for_obstacle) decides when it's actually close enough.
    xmin, ymin, xmax, ymax = bbox
    width = max(0, xmax - xmin)
    height = max(0, ymax - ymin)
    if width <= 2 or height <= 2:
        return False
    return width * height > 600
