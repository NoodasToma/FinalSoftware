# 🚦 Stashed: simulation traffic lights (removed for now)

The traffic light was removed from the **Godot simulation** scene (it was hard to
trigger reliably in the compressed sim layout — see `tasks/project/SIM_TESTS.md`).
The agent's traffic-light *capability* and its headless tests are **untouched**;
only the in-sim scene objects + the sim demo station were removed. Everything
needed to put it back is here.

## What was removed from `GodotSimulation/ducky-bot/scenes/maps/project.tscn`

**ext_resources** (header):
```
[ext_resource type="PackedScene" path="res://scenes/objects/obj_traffic_light.tscn" id="16_tlight"]
[ext_resource type="Texture2D" path="res://textures/tag36h11/tag36_11_00074.png" id="18_tag74"]
```

**Nodes**:
```
[node name="Sign_tlight" parent="Signs" instance=ExtResource("13_tagbb")]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 6.0, 0, 3.2)
sign_texture = ExtResource("18_tag74")

[node name="Label" type="Label3D" parent="Signs/Sign_tlight"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.4, 0)
pixel_size = 0.0028
billboard = 1
double_sided = true
modulate = Color(1, 1, 1, 1)
outline_modulate = Color(0, 0, 0, 1)
outline_size = 16
font_size = 30
text = "t-light-ahead"

[node name="TrafficLight_demo" parent="." instance=ExtResource("16_tlight")]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 6.05, 0, 2.7)

[node name="Line_Light" type="MeshInstance3D" parent="StopLines"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 5.85, 0.096, 3.1)
mesh = SubResource("StopLineNS")
```

## Stashed asset files (moved out of the Godot project tree)

- `obj_traffic_light.tscn`  ← was `GodotSimulation/ducky-bot/scenes/objects/`
- `traffic_light.gd`        ← was `GodotSimulation/ducky-bot/scripts/`
  (the `.tscn` references the script by `res://scripts/traffic_light.gd`)

## What was NOT touched (capability still exists)

- `tasks/project/packages/perception/traffic_light.py` — the HSV detector.
- `tasks/project/packages/agent.py` — arms the detector on a `t-light-ahead`
  AprilTag (ids 74–94, 200–230) and does stop-on-red / go-on-green. With no
  t-light sign in the sim scene it simply never arms — harmless.
- `tasks/project/tests/` — the unit + integration traffic-light tests still pass.

## To re-enable in the sim

1. Move the two files back:
   - `obj_traffic_light.tscn` → `GodotSimulation/ducky-bot/scenes/objects/`
   - `traffic_light.gd` → `GodotSimulation/ducky-bot/scripts/`
2. Paste the ext_resources + the four nodes above back into `project.tscn`
   (the nodes go in the `Signs` / root / `StopLines` parents as shown).
3. (Optional) restore the dashboard's traffic-light demo station + `_light`
   debug tile in `servers/project/virtual_server.py` (removed when the light was
   stashed — see that file's git history for this change).
