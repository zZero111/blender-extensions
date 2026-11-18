# SPDX-License-Identifier: GPL-3.0-or-later
"""Export Fixer: find and fix mirrored, non-uniform and unapplied scale before export."""

import bpy

from . import operators, overlay, props, ui

_classes = props.classes + operators.classes + ui.classes


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.export_fixer = bpy.props.PointerProperty(type=props.ExportFixerProps)
    bpy.types.TOPBAR_MT_file_export.append(ui.draw_export_menu)
    if not bpy.app.background:
        overlay.enable()


def unregister():
    overlay.disable()
    bpy.types.TOPBAR_MT_file_export.remove(ui.draw_export_menu)
    del bpy.types.Scene.export_fixer
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
