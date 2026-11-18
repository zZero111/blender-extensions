# SPDX-License-Identifier: GPL-3.0-or-later
"""Export targets and the exporter settings each one uses."""

PRESETS = {
    "UNITY": {
        "label": "Unity (FBX)",
        "format": "FBX",
        "ext": ".fbx",
        "summary": "-Z fwd, Y up, FBX All, baked",
        "kwargs": {
            "apply_unit_scale": True,
            "apply_scale_options": "FBX_SCALE_ALL",
            "axis_forward": "-Z",
            "axis_up": "Y",
            "bake_space_transform": True,
            "mesh_smooth_type": "FACE",
            "add_leaf_bones": False,
        },
    },
    "UNREAL": {
        "label": "Unreal (FBX)",
        "format": "FBX",
        "ext": ".fbx",
        "summary": "FBX Units, face smoothing",
        "kwargs": {
            "apply_unit_scale": True,
            "apply_scale_options": "FBX_SCALE_UNITS",
            "axis_forward": "-Z",
            "axis_up": "Y",
            "mesh_smooth_type": "FACE",
            "add_leaf_bones": False,
        },
    },
    "GODOT": {
        "label": "Godot (glTF .glb)",
        "format": "GLTF",
        "ext": ".glb",
        "summary": "GLB, +Y up, modifiers on",
        "kwargs": {
            "export_format": "GLB",
            "export_yup": True,
            "export_apply": True,
        },
    },
    "GLTF": {
        "label": "glTF / Web (.glb)",
        "format": "GLTF",
        "ext": ".glb",
        "summary": "GLB, +Y up, modifiers on",
        "kwargs": {
            "export_format": "GLB",
            "export_yup": True,
            "export_apply": True,
        },
    },
    "FBX_DEFAULT": {
        "label": "FBX (Blender defaults)",
        "format": "FBX",
        "ext": ".fbx",
        "summary": "Blender FBX defaults",
        "kwargs": {},
    },
}


def enum_items():
    return [(key, p["label"], p["summary"]) for key, p in PRESETS.items()]
