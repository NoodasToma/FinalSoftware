extends Node3D
# Cycling traffic light for the `project` simulation.
#
# Drives an UNSHADED coloured lens through red -> green -> yellow on a timer so
# the bot's HSV detector (tasks/project/packages/perception/traffic_light.py)
# can see it. Unshaded means the rendered pixel == the albedo colour regardless
# of scene lighting, so the colours land squarely inside the detector's HSV
# bands. The lens sits high on a post so it falls in the TOP HALF of the camera
# frame, which is the only region the detector searches.
#
# Tunable per instance from the scene (e.g. stagger two lights with start_offset).

@export var red_seconds: float = 7.0
@export var green_seconds: float = 6.0
@export var yellow_seconds: float = 2.5
@export var start_offset: float = 0.0   # phase shift, so multiple lights differ

# Saturated, bright colours - chosen to pass the detector's HSV thresholds even
# after the sim's double JPEG compression (Godot q0.75 -> server q0.70).
const C_RED := Color(0.95, 0.0, 0.0)
const C_GREEN := Color(0.0, 0.85, 0.10)
const C_YELLOW := Color(1.0, 0.82, 0.0)

var _t: float = 0.0
var _mat: StandardMaterial3D = null

func _ready() -> void:
	_t = start_offset
	var mi := get_node_or_null("Lens") as MeshInstance3D
	if mi == null:
		push_warning("[TrafficLight] no 'Lens' MeshInstance3D child")
		return
	# Own material per instance so two lights animate independently.
	_mat = StandardMaterial3D.new()
	_mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mi.material_override = _mat
	_apply()

func _process(delta: float) -> void:
	_t += delta
	_apply()

func current_color() -> Color:
	var cycle: float = red_seconds + green_seconds + yellow_seconds
	if cycle <= 0.0:
		return C_RED
	var p: float = fmod(_t, cycle)
	if p < red_seconds:
		return C_RED
	elif p < red_seconds + green_seconds:
		return C_GREEN
	return C_YELLOW

func _apply() -> void:
	if _mat != null:
		_mat.albedo_color = current_color()
