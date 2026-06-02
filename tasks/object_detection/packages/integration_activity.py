from typing import Tuple

# Path to the trained model weights (.onnx file).
# Relative paths resolve from the project root.
MODEL_PATH = "tasks/object_detection/models/best.onnx"


def NUMBER_FRAMES_SKIPPED() -> int:
    # Higher = run inference less often (cheaper).
    return 1


def filter_by_classes(pred_class: int) -> bool:
    """Return False to drop this prediction."""
    # Only duckies (class 0) are pedestrians the bot must avoid; trucks and
    # signs are off the road, so drop them.
    return pred_class == 0


def filter_by_scores(score: float) -> bool:
    """Confidence in [0.0, 1.0]. Return False to drop low-confidence boxes."""
    return score >= 0.6


def filter_by_bboxes(bbox: Tuple[int, int, int, int]) -> bool:
    """bbox is (xmin, ymin, xmax, ymax) in pixels. Return False to drop."""
    # Drop tiny boxes (far-off duckies not in the bot's path).
    xmin, ymin, xmax, ymax = bbox
    area = (xmax - xmin) * (ymax - ymin)
    return area > 800
