# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure detection logic: what is wrong with an object's transform."""

EPS = 1e-4

# Codes
NEGATIVE = "NEGATIVE"
NON_UNIFORM = "NON_UNIFORM"
UNAPPLIED_SCALE = "UNAPPLIED_SCALE"
UNAPPLIED_ROT = "UNAPPLIED_ROT"
ZERO_SCALE = "ZERO_SCALE"
UNIT_SCALE = "UNIT_SCALE"

SEVERITY_RANK = {"ERROR": 3, "WARNING": 2, "INFO": 1}


def _local_scale(ob):
    """Scale including delta scale, as a list of three floats."""
    return [ob.scale[i] * ob.delta_scale[i] for i in range(3)]


def _is_uniform(values):
    a = [abs(v) for v in values]
    return max(a) - min(a) <= EPS * max(max(a), 1.0)


def _is_editable(ob):
    if ob.library is not None:
        return False
    data = ob.data
    return data is None or data.library is None


def object_problems(ob, opts):
    """Return a list of (code, severity, message, fixable) for one object.

    `opts` is the scene's ExportFixerProps (check_* toggles).
    Only mesh objects are checked; other types are ignored.
    """
    if ob.type != "MESH":
        return []

    out = []
    editable = _is_editable(ob)
    local = _local_scale(ob)
    world = ob.matrix_world.to_scale()

    if any(abs(v) < 1e-8 for v in local):
        return [(ZERO_SCALE, "ERROR", "Zero scale on an axis: object is flat and can't be applied", False)]

    local_negative = (local[0] * local[1] * local[2]) < 0
    world_negative = ob.matrix_world.is_negative
    parent_name = ob.parent.name if ob.parent else ""

    if opts.check_negative and world_negative:
        if local_negative:
            out.append((NEGATIVE, "ERROR", "Mirrored (negative scale): faces flip in most engines", editable))
        else:
            out.append((NEGATIVE, "ERROR", f"Mirrored by parent '{parent_name}': fix the parent", False))

    if opts.check_non_uniform and not _is_uniform(world):
        if not _is_uniform(local):
            out.append((NON_UNIFORM, "WARNING", "Non-uniform scale: bevel, inset and physics distort", editable))
        else:
            out.append((NON_UNIFORM, "WARNING", f"Non-uniform scale inherited from '{parent_name}'", False))

    already = {c for c, *_ in out}
    scaled = any(abs(abs(v) - 1.0) > EPS for v in local)
    if opts.check_unapplied_scale and scaled and not ({NEGATIVE, NON_UNIFORM} & already):
        msg = f"Unapplied scale {abs(local[0]):.3g}: size will differ in engine"
        out.append((UNAPPLIED_SCALE, "WARNING", msg, editable))

    if opts.check_rotation:
        angle = ob.matrix_basis.to_quaternion().angle
        if angle > EPS:
            out.append((UNAPPLIED_ROT, "INFO", "Unapplied rotation", editable))

    return out


def scene_problems(scene, opts):
    out = []
    us = scene.unit_settings
    if opts.check_unit_scale and abs(us.scale_length - 1.0) > EPS:
        out.append((UNIT_SCALE, "WARNING", f"Scene unit scale is {us.scale_length:g}, engines expect 1.0", True))
    return out


def objects_to_check(context, selected_only):
    if selected_only:
        return list(context.selected_objects)
    return [ob for ob in context.view_layer.objects if ob.visible_get()]
