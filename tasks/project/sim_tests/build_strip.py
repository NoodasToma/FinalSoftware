"""Augment the proven `project.tscn` loop with the full sign-test bays.

Idempotent in-place surgery on GodotSimulation/ducky-bot/scenes/maps/project.tscn:
  1. add Texture2D ext_resources for every tag the bays need (skipped if present);
  2. enlarge the Ground plane + collision so the bay field (z up to ~19) is covered;
  3. (re)append a `SignBays` node (one bay per turn-constraint sign) and a
     `DecodeSigns` node (roadside decode-only signs) below an AUTO marker.

The loop (lane / 4-way / light / obstacle / vehicle / ducks / NPC) is left intact,
so behaviour_suite.py and the continuous course keep working. Geometry + tag
assignments come from course_map.py, so the scene and the tests never drift.

Run:  .venv311\Scripts\python.exe tasks\project\sim_tests\build_strip.py
"""
from __future__ import annotations
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _ROOT)

from tasks.project.sim_tests.course_map import (
    SIGN_BAYS, DECODE_SIGNS, bay_geom,
)
from tasks.project.packages.sign_registry import lookup

SCENE = os.path.join(_ROOT, "GodotSimulation", "ducky-bot", "scenes", "maps", "project.tscn")
MARKER = "; ===== AUTO-GENERATED SIGN BAYS (build_strip.py) ====="
TILE = '1_xdp4q'          # tile_straight, identity = N-S road
BILLBOARD = '13_tagbb'    # obj_tag_billboard.tscn (sign.gd, sign_texture export)
STOPLINE = 'StopLineNS'   # sub_resource thin red BoxMesh already in project.tscn

# Tags already declared in the loop scene -> reuse their ext_resource id.
EXISTING_TAG_IDS = {1: '7_xv6rw', 8: '17_tag8', 74: '18_tag74', 400: '15_tag400'}


def _tex_path(tag: int) -> str:
    return f"res://textures/tag36h11/tag36_11_{tag:05d}.png"


def _tag_extid(tag: int) -> str:
    return EXISTING_TAG_IDS.get(tag, f"bay_tag{tag}")


def _ident(x, y, z) -> str:
    return f"Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, {x}, {y}, {z})"


def _needed_tags() -> list[int]:
    tags = set()
    for _name, _cell, trig, test, _legal in SIGN_BAYS:
        tags.add(trig)
        if test is not None:
            tags.add(test)
    for _x, _z, tag, _n in DECODE_SIGNS:
        tags.add(tag)
    return sorted(tags)


def _ext_lines() -> list[str]:
    """ext_resource lines for tags not already in the scene."""
    out = []
    for tag in _needed_tags():
        if tag in EXISTING_TAG_IDS:
            continue
        out.append(f'[ext_resource type="Texture2D" path="{_tex_path(tag)}" '
                   f'id="{_tag_extid(tag)}"]')
    return out


def _meaning(tag: int) -> str:
    """Human label for a tag (e.g. 'no-left-turn', 'stop', 'vehicle')."""
    sem = lookup(tag)
    if sem is None:
        return f"tag {tag}"
    return sem.kind or (sem.tag_type.lower() if sem.tag_type else f"tag {tag}")


def _label(node_name: str, parent: str, text: str) -> str:
    """A readable Label3D floating above a sign so a human can tell signs apart.
    White text + black outline (white is outside the light detector's HSV bands,
    so it can't be mistaken for a traffic-light colour); sits well above the tag
    so it never occludes the AprilTag for the camera."""
    return (f'\n[node name="{node_name}" type="Label3D" parent="{parent}"]\n'
            f'transform = {_ident(0, 0.4, 0)}\n'
            f'pixel_size = 0.0028\n'
            f'billboard = 1\n'
            f'double_sided = true\n'
            f'modulate = Color(1, 1, 1, 1)\n'
            f'outline_modulate = Color(0, 0, 0, 1)\n'
            f'outline_size = 16\n'
            f'font_size = 30\n'
            f'text = "{text}"\n')


def _billboard(node_name: str, parent: str, x, y, z, tag: int) -> str:
    return (f'\n[node name="{node_name}" parent="{parent}" instance=ExtResource("{BILLBOARD}")]\n'
            f'transform = {_ident(x, y, z)}\n'
            f'sign_texture = ExtResource("{_tag_extid(tag)}")\n')


def _tile(node_name: str, parent: str, x, z) -> str:
    return (f'\n[node name="{node_name}" parent="{parent}" instance=ExtResource("{TILE}")]\n'
            f'transform = {_ident(x, 0.091, z)}\n')


def _line(node_name: str, parent: str, x, z) -> str:
    return (f'\n[node name="{node_name}" type="MeshInstance3D" parent="{parent}"]\n'
            f'transform = {_ident(x, 0.096, z)}\n'
            f'mesh = SubResource("{STOPLINE}")\n')


def _bays_block() -> str:
    s = '\n[node name="SignBays" type="Node3D" parent="."]\n'
    for name, cell, trig, test, _legal in SIGN_BAYS:
        g = bay_geom(*cell)
        pre = f"Bay_{name}"
        for i, tz in enumerate(g["tile_zs"]):
            s += _tile(f"{pre}_tile{i}", "SignBays", g["x"], tz)
        s += _line(f"{pre}_line", "SignBays", g["line_x"], g["line_z"])
        s += _billboard(f"{pre}_trig", "SignBays", g["trigger_x"], 0, g["sign_z"], trig)
        s += _label(f"{pre}_trig_lbl", f"SignBays/{pre}_trig", _meaning(trig))
        if test is not None:
            s += _billboard(f"{pre}_test", "SignBays", g["test_x"], 0, g["sign_z"], test)
            s += _label(f"{pre}_test_lbl", f"SignBays/{pre}_test", _meaning(test))
    return s


def _decode_block() -> str:
    s = '\n[node name="DecodeSigns" type="Node3D" parent="."]\n'
    for x, z, tag, name in DECODE_SIGNS:
        s += _billboard(f"Decode_{name}", "DecodeSigns", x, 0, z, tag)
        s += _label(f"Decode_{name}_lbl", f"DecodeSigns/Decode_{name}", _meaning(tag))
    return s


def main() -> int:
    with open(SCENE, encoding="utf-8") as fh:
        text = fh.read()

    # 1) strip any previous auto block (idempotent)
    if MARKER in text:
        text = text[:text.index(MARKER)].rstrip() + "\n"

    # 2) inject ext_resources after the last existing ext_resource line
    lines = text.splitlines()
    last_ext = max(i for i, ln in enumerate(lines) if ln.startswith("[ext_resource"))
    inject = [ln for ln in _ext_lines() if ln.split('path="')[1].split('"')[0] not in text]
    if inject:
        lines[last_ext + 1:last_ext + 1] = inject
    text = "\n".join(lines)

    # 3) enlarge the ground so the bay field is covered
    text = (text
            .replace("size = Vector3(28.8, 0.1, 28.8)", "size = Vector3(44, 0.1, 44)")
            .replace("size = Vector2(28.8, 28.8)", "size = Vector2(44, 44)")
            .replace("1, 4.8, 0.041, 4.8)", "1, 6, 0.041, 10)")
            .replace("1, 4.8, 0.084, 4.8)", "1, 6, 0.084, 10)"))

    # 4) append the auto block
    text = text.rstrip() + "\n\n" + MARKER + "\n" + _bays_block() + _decode_block()

    with open(SCENE, "w", encoding="utf-8") as fh:
        fh.write(text)

    nbays = len(SIGN_BAYS)
    ndec = len(DECODE_SIGNS)
    print(f"OK: wrote {SCENE}")
    print(f"  +{len(inject)} tag ext_resources, {nbays} sign bays, {ndec} decode signs")
    print(f"  tags used: {_needed_tags()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
