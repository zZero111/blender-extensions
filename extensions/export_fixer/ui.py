# SPDX-License-Identifier: GPL-3.0-or-later
import textwrap

import bpy

from . import presets

SEVERITY_ICON = {"ERROR": "CANCEL", "WARNING": "ERROR", "INFO": "INFO"}


class EXPORTFIXER_UL_issues(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        label = item.obj_name or "Scene"
        row.label(text=label, icon=SEVERITY_ICON[item.severity])
        if item.obj_name:
            op = row.operator("object.export_fixer_select", text="", icon="RESTRICT_SELECT_OFF", emboss=False)
            op.index = index
        sub = row.row(align=True)
        sub.enabled = item.fixable
        op = sub.operator("object.export_fixer_fix", text="", icon="CHECKMARK", emboss=False)
        op.index = index


class VIEW3D_PT_export_fixer(bpy.types.Panel):
    bl_label = "Export Fixer"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Export Fixer"

    def draw(self, context):
        layout = self.layout
        opts = context.scene.export_fixer

        col = layout.column(align=True)
        col.prop(opts, "target", text="")
        col.label(text=presets.PRESETS[opts.target]["summary"])

        col = layout.column()
        col.prop(opts, "selected_only")
        col.prop(opts, "show_overlay")

        layout.operator("object.export_fixer_scan", icon="VIEWZOOM")

        if opts.scanned:
            errors = sum(1 for i in opts.issues if i.severity == "ERROR")
            warns = sum(1 for i in opts.issues if i.severity == "WARNING")
            box = layout.box()
            if not opts.issues:
                box.label(text="All clear, ready to export", icon="CHECKMARK")
            else:
                box.label(text=f"{errors} error(s), {warns} warning(s)", icon="ERROR" if errors else "INFO")
                box.template_list("EXPORTFIXER_UL_issues", "", opts, "issues", opts, "issue_index", rows=4)
                if 0 <= opts.issue_index < len(opts.issues):
                    msg = opts.issues[opts.issue_index].message
                    # Wrap to the sidebar width so long messages aren't cut off.
                    scale = context.preferences.system.ui_scale
                    chars = max(20, int(context.region.width / (6.5 * scale)))
                    col = box.column(align=True)
                    for line in textwrap.wrap(msg, chars):
                        col.label(text=line)
                box.operator("object.export_fixer_fix_all", icon="CHECKMARK")

        layout.separator()
        layout.operator("export_scene.export_fixer_export", icon="EXPORT")


class VIEW3D_PT_export_fixer_settings(bpy.types.Panel):
    bl_label = "Settings"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Export Fixer"
    bl_parent_id = "VIEW3D_PT_export_fixer"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        opts = context.scene.export_fixer
        col = layout.column(heading="Check")
        col.prop(opts, "check_negative")
        col.prop(opts, "check_non_uniform")
        col.prop(opts, "check_unapplied_scale")
        col.prop(opts, "check_rotation")
        col.prop(opts, "check_unit_scale")
        layout.separator()
        layout.prop(opts, "fix_includes_rotation")
        layout.prop(opts, "block_on_errors")


def draw_export_menu(self, context):
    self.layout.operator("export_scene.export_fixer_export", text="Export Fixer: Check & Export (.fbx/.glb)")


classes = (EXPORTFIXER_UL_issues, VIEW3D_PT_export_fixer, VIEW3D_PT_export_fixer_settings)
