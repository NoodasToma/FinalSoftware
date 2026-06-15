from typing import List

from tasks.project.packages.sign_registry import SignSemantic


def we_go_first(my_name: str, recent_vehicle_signs: List[SignSemantic]) -> bool:
    for sign in recent_vehicle_signs:
        if sign.tag_type != "Vehicle" or sign.vehicle_name is None:
            continue
        if sign.vehicle_name < my_name:
            return False
    return True
