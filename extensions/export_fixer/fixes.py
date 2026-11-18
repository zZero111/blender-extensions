# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixes that change data. All work without bpy.ops, so they run in any context."""

import bmesh
from mathutils import Matrix


def make_single_user(ob):
    """Give the object its own copy of the mesh if other objects share it.

    Returns True if a copy was made.
    """
    data = ob.data
    real_users = data.users - (1 if data.use_fake_user else 0)
    if real_users > 1:
        ob.data = data.copy()
        return True
    return False


def _reference_face(mesh):
    """Index of the first face with real area (a stable face to compare normals on)."""
    for poly in mesh.polygons:
        if poly.area > 1e-12:
            return poly.index
    return None


def _reverse_all_faces(mesh):
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bmesh.ops.reverse_faces(bm, faces=bm.faces[:], flip_multires=True)
        bm.to_mesh(mesh)
    finally:
        bm.free()
    mesh.update()


def apply_transform(ob, include_rotation=False):
    """Bake scale (and optionally rotation) into the mesh, keeping it looking the same.

    - Handles delta transforms.
    - Keeps children where they are.
    - If the object was mirrored, makes sure normals still point outward.
    Returns a short note describing what was done.
    """
    notes = []
    if make_single_user(ob):
        notes.append("made mesh single-user")

    if include_rotation:
        linear = ob.matrix_basis.to_3x3()
    else:
        s = [ob.scale[i] * ob.delta_scale[i] for i in range(3)]
        linear = Matrix.Diagonal(s)

    if abs(linear.determinant()) < 1e-12:
        return "skipped: zero scale"

    mesh = ob.data
    ref = _reference_face(mesh)
    expected = None
    if ref is not None:
        expected = linear.inverted().transposed() @ mesh.polygons[ref].normal

    m4 = linear.to_4x4()
    mesh.transform(m4, shape_keys=True)
    mesh.update()

    # Reset the object's own channels.
    ob.scale = (1.0, 1.0, 1.0)
    ob.delta_scale = (1.0, 1.0, 1.0)
    if include_rotation:
        ob.rotation_euler = (0.0, 0.0, 0.0)
        ob.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        ob.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
        ob.delta_rotation_euler = (0.0, 0.0, 0.0)
        ob.delta_rotation_quaternion = (1.0, 0.0, 0.0, 0.0)

    # Keep children in place: parent world loses `m4`, so their parent-inverse gains it.
    for child in ob.children:
        child.matrix_parent_inverse = m4 @ child.matrix_parent_inverse

    # Mirrored meshes: vertex positions flip but face winding does not,
    # so normals end up pointing inward. Check against the expected normal.
    if expected is not None and linear.determinant() < 0:
        n_after = mesh.polygons[ref].normal
        if n_after.dot(expected) < 0:
            _reverse_all_faces(mesh)
            notes.append("flipped normals outward")

    notes.insert(0, "applied scale and rotation" if include_rotation else "applied scale")
    return ", ".join(notes)
