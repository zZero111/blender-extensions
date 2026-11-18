# SPDX-License-Identifier: GPL-3.0-or-later
import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)

from . import presets


class ExportFixerIssue(bpy.types.PropertyGroup):
    obj: PointerProperty(type=bpy.types.Object)  # None for scene-level issues
    obj_name: StringProperty()  # display only; empty for scene-level issues
    code: StringProperty()
    message: StringProperty()
    severity: EnumProperty(
        items=[
            ("ERROR", "Error", ""),
            ("WARNING", "Warning", ""),
            ("INFO", "Info", ""),
        ],
        default="WARNING",
    )
    fixable: BoolProperty(default=False)


def _redraw_viewports(self, context):
    for area in context.screen.areas if context.screen else []:
        if area.type == "VIEW_3D":
            area.tag_redraw()


class ExportFixerProps(bpy.types.PropertyGroup):
    target: EnumProperty(
        name="Target",
        description="Engine or format you are exporting to",
        items=presets.enum_items(),
        default="UNITY",
    )
    selected_only: BoolProperty(
        name="Selected Only",
        description="Check and export only selected objects",
        default=True,
    )
    check_negative: BoolProperty(name="Mirrored (negative) scale", default=True)
    check_non_uniform: BoolProperty(name="Non-uniform scale", default=True)
    check_unapplied_scale: BoolProperty(name="Unapplied scale", default=True)
    check_rotation: BoolProperty(
        name="Unapplied rotation",
        description="Off by default: many pipelines keep object rotation on purpose",
        default=False,
    )
    check_unit_scale: BoolProperty(name="Scene unit scale", default=True)
    fix_includes_rotation: BoolProperty(
        name="Fixes Also Apply Rotation",
        description="When fixing scale, also bake rotation into the mesh",
        default=False,
    )
    block_on_errors: BoolProperty(
        name="Block Export on Errors",
        description="Refuse to export while any error-level issue remains",
        default=True,
    )
    show_overlay: BoolProperty(
        name="Highlight in Viewport",
        description="Draw coloured boxes around objects with problems",
        default=True,
        update=_redraw_viewports,
    )
    scanned: BoolProperty(default=False)
    issues: CollectionProperty(type=ExportFixerIssue)
    issue_index: IntProperty()


classes = (ExportFixerIssue, ExportFixerProps)
