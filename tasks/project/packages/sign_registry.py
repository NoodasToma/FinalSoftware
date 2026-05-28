

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


_ALL_TURNS: frozenset[str] = frozenset({"left", "right", "straight"})

_TURN_MAP: dict[str, frozenset[str]] = {
    "4-way-intersect":  frozenset({"left", "right", "straight"}),
    "T-intersection":   frozenset({"left", "right"}),
    "right-T-intersect": frozenset({"straight", "right"}),
    "left-T-intersect": frozenset({"straight", "left"}),
    "oneway-right":     frozenset({"right"}),
    "oneway-left":      frozenset({"left"}),
    "do-not-enter":     frozenset(),
    # everything else -> no constraint (full set)
}

@dataclass(frozen=True)
class SignSemantic:

    kind: str          # traffic_sign_type
    tag_type: str
    vehicle_name: Optional[str]

    @property
    def available_turns(self) -> set[str]:
        """Return the set of legal turns implied by this sign.

        Uses the closed mapping from the spec; any ``kind`` not explicitly
        listed returns the full set (no constraint).
        """
        return set(_TURN_MAP.get(self.kind, _ALL_TURNS))


def _find_db_path() -> str:
    """Locate apriltagsDB.yaml relative to this file."""
    # This file lives at tasks/project/packages/sign_registry.py
    # The DB is at  tasks/project/packages/apriltagsDB.yaml
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "apriltagsDB.yaml"),
        # fallback: one level up (shouldn't be needed but keeps tests easy)
        os.path.join(here, "..", "apriltagsDB.yaml"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.normpath(path)
    raise FileNotFoundError(
        "apriltagsDB.yaml not found. Searched:\n  " + "\n  ".join(candidates)
    )


def _load_db() -> dict[int, SignSemantic]:
    """Parse the YAML file and return a tag_id → SignSemantic dict."""
    import yaml  # PyYAML — available on the bot image

    db_path = _find_db_path()
    with open(db_path) as fh:
        entries = yaml.safe_load(fh)

    if not isinstance(entries, list):
        raise ValueError(f"apriltagsDB.yaml: expected a list, got {type(entries)}")

    registry: dict[int, SignSemantic] = {}
    skipped = 0

    for entry in entries:
        tag_id = entry.get("tag_id")
        if tag_id is None:
            skipped += 1
            continue

        tag_id = int(tag_id)

        # Coerce None -> '' for string fields so callers never get None for kind/tag_type
        tag_type = entry.get("tag_type") or ""
        kind = entry.get("traffic_sign_type") or ""
        vehicle_name_raw = entry.get("vehicle_name")
        vehicle_name = str(vehicle_name_raw) if vehicle_name_raw else None

        registry[tag_id] = SignSemantic(
            kind=kind,
            tag_type=tag_type,
            vehicle_name=vehicle_name,
        )

    logger.info(
        "sign_registry: loaded %d entries from %s  (skipped %d malformed)",
        len(registry),
        db_path,
        skipped,
    )
    return registry


# Module-level singleton — loaded exactly once.
try:
    _REGISTRY: dict[int, SignSemantic] = _load_db()
except Exception as _exc:  # noqa: BLE001
    logger.error("sign_registry: failed to load DB — %s", _exc)
    _REGISTRY = {}


def lookup(tag_id: int) -> Optional[SignSemantic]:
   
    return _REGISTRY.get(tag_id)
