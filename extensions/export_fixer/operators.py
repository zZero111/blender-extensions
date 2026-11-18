# SPDX-License-Identifier: GPL-3.0-or-later
import os

import bpy
from bpy.props import BoolProperty, IntProperty, StringProperty

from . import checks, fixes, presets

# ---------------------------------------------------------------- helpers

def run_scan(context):
    """Fill scene.export_fixer.issues. Returns (errors, warnings, infos)."""
    scene = context.scene
    opts = scene.export_fixer
    # Fixes change transforms; make sure matrix_world is re-evaluated before checking.
    context.view_layer.update()
    opts.issues.clear()

    rows = []
    for code, sev, msg, fixable in checks.scene_problems(scene, opts):
        rows.append((None, code, sev, msg, fixable))
    for ob in checks.objects_to_check(context, opts.selected_only):
        for code, sev, msg, fixable in checks.object_problems(ob, opts):
            rows.append((ob, code, sev, msg, fixable))

    rows.sort(key=lambda r: (-checks.SEVERITY_RANK[r[2]], r[0].name if r[0] else ""))
    counts = {"ERROR": 0, "WARNING": 0, "INFO": 0}
    for ob, code, sev, msg, fixable in rows:
        item = opts.issues.add()
        item.obj = ob
        item.obj_name = ob.name if ob else ""
        item.code = code
        item.severity = sev
        item.message = msg
        item.fixable = fixable
        counts[sev] += 1

    opts.issue_index = 0
    opts.scanned = True
    _tag_redraw(context)
    return counts["ERROR"], counts["WARNING"], counts["INFO"]


def _tag_redraw(context):
    if context.screen:
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _ensure_object_mode(context):
    if context.mode != "OBJECT" and context.active_object is not None:
        bpy.ops.object.mode_set(mode="OBJECT")


def fix_issue(context, item):
    """Apply the fix for one issue row. Returns a note string, or None if nothing done."""
    opts = context.scene.export_fixer
    if not item.fixable:
        return None
    if item.code == checks.UNIT_SCALE:
        context.scene.unit_settings.scale_length = 1.0
        return "Scene unit scale set to 1.0"

    ob = item.obj
    if ob is None or ob.type != "MESH":
        return None
    include_rot = opts.fix_includes_rotation or item.code == checks.UNAPPLIED_ROT
    return f"{ob.name}: " + fixes.apply_transform(ob, include_rotation=include_rot)


def _summary(e, w, i):
    if e == w == i == 0:
        return "Export Fixer: all clear"
    return f"Export Fixer: {e} error(s), {w} warning(s), {i} info"


# ---------------------------------------------------------------- operators

class OBJECT_OT_export_fixer_scan(bpy.types.Operator):
    """Check objects for transform problems that break exports"""

    bl_idname = "object.export_fixer_scan"
    bl_label = "Run Check"
    bl_options = {"REGISTER"}

    def execute(self, context):
        e, w, i = run_scan(context)
        self.report({"WARNING"} if e or w else {"INFO"}, _summary(e, w, i))
        return {"FINISHED"}


class OBJECT_OT_export_fixer_select(bpy.types.Operator):
    """Select this object and make it active"""

    bl_idname = "object.export_fixer_select"
    bl_label = "Select Object"
    bl_options = {"REGISTER", "UNDO"}

    index: IntProperty(options={"HIDDEN", "SKIP_SAVE"})

    def execute(self, context):
        issues = context.scene.export_fixer.issues
        ob = issues[self.index].obj if 0 <= self.index < len(issues) else None
        if ob is None or ob.name not in context.view_layer.objects:
            self.report({"WARNING"}, "That object is no longer in this view layer, run the check again")
            return {"CANCELLED"}
        _ensure_object_mode(context)
        for other in context.selected_objects:
            other.select_set(False)
        ob.hide_set(False)
        ob.select_set(True)
        context.view_layer.objects.active = ob
        return {"FINISHED"}


class OBJECT_OT_export_fixer_fix(bpy.types.Operator):
    """Fix this issue (applies scale; mirrored objects keep their normals outward)"""

    bl_idname = "object.export_fixer_fix"
    bl_label = "Fix Issue"
    bl_options = {"REGISTER", "UNDO"}

    index: IntProperty(options={"HIDDEN", "SKIP_SAVE"})

    def execute(self, context):
        opts = context.scene.export_fixer
        if not (0 <= self.index < len(opts.issues)):
            return {"CANCELLED"}
        _ensure_object_mode(context)
        note = fix_issue(context, opts.issues[self.index])
        run_scan(context)
        if note:
            self.report({"INFO"}, note)
            return {"FINISHED"}
        self.report({"WARNING"}, "Nothing to fix here")
        return {"CANCELLED"}


class OBJECT_OT_export_fixer_fix_all(bpy.types.Operator):
    """Fix every fixable issue in the list"""

    bl_idname = "object.export_fixer_fix_all"
    bl_label = "Fix All"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return any(i.fixable for i in context.scene.export_fixer.issues)

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        _ensure_object_mode(context)
        opts = context.scene.export_fixer
        # One fix per object is enough: applying the transform clears all its scale issues.
        done_objects = set()
        fixed = 0
        for item in list(opts.issues):
            if not item.fixable:
                continue
            key = item.obj if item.obj is not None else item.code
            if key in done_objects:
                continue
            if fix_issue(context, item):
                fixed += 1
                done_objects.add(key)
        e, w, i = run_scan(context)
        self.report({"INFO"}, f"Fixed {fixed} item(s). " + _summary(e, w, i))
        return {"FINISHED"}


class EXPORT_OT_export_fixer_export(bpy.types.Operator):
    """Run the check, then export with the target's settings"""

    bl_idname = "export_scene.export_fixer_export"
    bl_label = "Check & Export"
    bl_options = {"REGISTER"}

    filepath: StringProperty(subtype="FILE_PATH", options={"SKIP_SAVE"})
    filter_glob: StringProperty(default="*.fbx;*.glb", options={"HIDDEN"})
    check_existing: BoolProperty(default=True, options={"HIDDEN"})

    def _blocked(self, context):
        opts = context.scene.export_fixer
        e, _w, _i = run_scan(context)
        if e and opts.block_on_errors:
            self.report(
                {"ERROR"},
                f"Export blocked: {e} error(s). Fix them, or turn off 'Block Export on Errors'",
            )
            return True
        return False

    def invoke(self, context, event):
        if self._blocked(context):
            return {"CANCELLED"}
        preset = presets.PRESETS[context.scene.export_fixer.target]
        if not self.filepath:
            blend = bpy.data.filepath
            base = os.path.splitext(blend)[0] if blend else "untitled"
            self.filepath = base + preset["ext"]
        self.filter_glob = "*" + preset["ext"]
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        if self._blocked(context):
            return {"CANCELLED"}
        opts = context.scene.export_fixer
        preset = presets.PRESETS[opts.target]
        path = bpy.path.ensure_ext(bpy.path.abspath(self.filepath), preset["ext"])

        if preset["format"] == "FBX":
            op = bpy.ops.export_scene.fbx
            name = "FBX"
        else:
            op = bpy.ops.export_scene.gltf
            name = "glTF"

        try:
            known = set(op.get_rna_type().properties.keys())
        except (AttributeError, KeyError):
            self.report({"ERROR"}, f"The {name} exporter isn't available. Enable it in Preferences > Add-ons")
            return {"CANCELLED"}

        kwargs = dict(preset["kwargs"])
        kwargs["filepath"] = path
        kwargs["use_selection"] = opts.selected_only
        dropped = [k for k in kwargs if k not in known]
        for k in dropped:
            del kwargs[k]

        _ensure_object_mode(context)
        result = op(**kwargs)
        if "FINISHED" not in result:
            self.report({"ERROR"}, f"{name} export failed, see the Info editor for details")
            return {"CANCELLED"}

        msg = f"Exported {os.path.basename(path)} for {preset['label']}"
        if dropped:
            msg += f" (this Blender ignored: {', '.join(dropped)})"
        self.report({"INFO"}, msg)
        return {"FINISHED"}


classes = (
    OBJECT_OT_export_fixer_scan,
    OBJECT_OT_export_fixer_select,
    OBJECT_OT_export_fixer_fix,
    OBJECT_OT_export_fixer_fix_all,
    EXPORT_OT_export_fixer_export,
)
