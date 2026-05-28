from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_ALL_TURNS: frozenset[str] = frozenset({"left", "right", "straight"})

_TURN_MAP: dict[str, frozenset[str]] = {
    "4-way-intersect":   frozenset({"left", "right", "straight"}),
    "T-intersection":    frozenset({"left", "right"}),
    "right-T-intersect": frozenset({"straight", "right"}),
    "left-T-intersect":  frozenset({"straight", "left"}),
    "oneway-right":      frozenset({"right"}),
    "oneway-left":       frozenset({"left"}),
    "do-not-enter":      frozenset(),
}


@dataclass(frozen=True)
class SignSemantic:
    kind: str
    tag_type: str
    vehicle_name: Optional[str]

    @property
    def available_turns(self) -> set[str]:
        return set(_TURN_MAP.get(self.kind, _ALL_TURNS))


def _find_db_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "apriltagsDB.yaml"),
        os.path.join(here, "..", "apriltagsDB.yaml"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.normpath(path)
    raise FileNotFoundError("apriltagsDB.yaml not found. Searched: " + str(candidates))


def _load_db() -> dict[int, SignSemantic]:
    import yaml

    with open(_find_db_path()) as fh:
        entries = yaml.safe_load(fh)

    if not isinstance(entries, list):
        raise ValueError(f"apriltagsDB.yaml: expected list, got {type(entries)}")

    registry: dict[int, SignSemantic] = {}
    for entry in entries:
        tag_id = entry.get("tag_id")
        if tag_id is None:
            continue
        vehicle_raw = entry.get("vehicle_name")
        registry[int(tag_id)] = SignSemantic(
            kind=entry.get("traffic_sign_type") or "",
            tag_type=entry.get("tag_type") or "",
            vehicle_name=str(vehicle_raw) if vehicle_raw else None,
        )

    logger.info("sign_registry: loaded %d entries", len(registry))
    return registry


try:
    _REGISTRY: dict[int, SignSemantic] = _load_db()
except Exception as _exc:
    logger.error("sign_registry: failed to load DB — %s", _exc)
    _REGISTRY = {}


def lookup(tag_id: int) -> Optional[SignSemantic]:
    return _REGISTRY.get(tag_id)
