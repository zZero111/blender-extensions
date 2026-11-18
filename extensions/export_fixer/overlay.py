# SPDX-License-Identifier: GPL-3.0-or-later
"""Viewport highlight: coloured bounding boxes around objects with problems.

Red = mirrored, orange = non-uniform, yellow = unapplied scale.
Computed live each redraw, so it updates as you edit without re-scanning.
Covers the same objects as the scan (selected or all visible), capped at MAX_OBJECTS.
"""

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from . import checks

COLORS = {
    checks.NEGATIVE: (1.0, 0.15, 0.15, 1.0),
    checks.ZERO_SCALE: (1.0, 0.15, 0.15, 1.0),
    checks.NON_UNIFORM: (1.0, 0.55, 0.1, 1.0),
    checks.UNAPPLIED_SCALE: (1.0, 0.9, 0.2, 1.0),
    checks.UNAPPLIED_ROT: (0.4, 0.7, 1.0, 1.0),
}
PRIORITY = [checks.NEGATIVE, checks.ZERO_SCALE, checks.NON_UNIFORM, checks.UNAPPLIED_SCALE, checks.UNAPPLIED_ROT]
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
MAX_OBJECTS = 2000

_handle = None
_shader = None
_warned = False


def _draw():
    global _shader, _warned
    if _warned:
        return
    try:
        context = bpy.context
        scene = context.scene
        opts = getattr(scene, "export_fixer", None)
        if opts is None or not opts.show_overlay:
            return
        if _shader is None:
            _shader = gpu.shader.from_builtin("UNIFORM_COLOR")

        # Same scope as the scan, capped so huge scenes stay responsive.
        candidates = checks.objects_to_check(context, opts.selected_only)
        meshes = [ob for ob in candidates if ob.type == "MESH" and ob.visible_get()][:MAX_OBJECTS]

        lines = {}
        for ob in meshes:
            codes = {p[0] for p in checks.object_problems(ob, opts)}
            code = next((c for c in PRIORITY if c in codes), None)
            if code is None:
                continue
            mw = ob.matrix_world
            corners = [mw @ Vector(c) for c in ob.bound_box]
            pts = lines.setdefault(code, [])
            for a, b in EDGES:
                pts.append(corners[a])
                pts.append(corners[b])

        if not lines:
            return
        gpu.state.depth_test_set("NONE")
        gpu.state.line_width_set(2.0)
        _shader.bind()
        for code, pts in lines.items():
            batch = batch_for_shader(_shader, "LINES", {"pos": pts})
            _shader.uniform_float("color", COLORS[code])
            batch.draw(_shader)
        gpu.state.line_width_set(1.0)
    except Exception as exc:  # noqa: BLE001
        # Never break the viewport; report the first failure once, then stay quiet.
        print(f"Export Fixer: viewport highlight disabled after error: {exc!r}")
        _warned = True


def enable():
    global _handle
    if _handle is None:
        _handle = bpy.types.SpaceView3D.draw_handler_add(_draw, (), "WINDOW", "POST_VIEW")


def disable():
    global _handle, _shader, _warned
    if _handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle, "WINDOW")
        _handle = None
    _shader = None
    _warned = False
