from __future__ import annotations

from tasks.project.packages.sign_registry import SignSemantic


def we_go_first(my_name: str, recent_vehicle_signs: list[SignSemantic]) -> bool:
    if not recent_vehicle_signs:
        return True

    for sign in recent_vehicle_signs:
        if sign.vehicle_name is not None and sign.vehicle_name < my_name:
            return False

    return True
