# SPDX-FileCopyrightText: 2020-2022 Blender Foundation
#
# SPDX-License-Identifier: GPL-2.0-or-later

bl_info = {
    "name": "Real Snow",
    "description": "Generate snow mesh",
    "author": "Marco Pavanello, Drew Perttula",
    "version": (1, 11, 1),
    "blender": (4, 2, 0),
    "location": "View 3D > Properties Panel",
    "doc_url": "{BLENDER_MANUAL_URL}/addons/object/real_snow.html",
    "tracker_url": "https://gitlab.com/marcopavanello/real-snow/-/issues",
    "support": "COMMUNITY",
    "category": "Object",
    }


# Libraries
import math
import os
import random

import numpy as np
import time

import bpy
import bmesh
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty
from bpy.types import Operator, Panel, PropertyGroup
from mathutils import Vector, noise
from mathutils.kdtree import KDTree
from mathutils.bvhtree import BVHTree


# Panel
class REAL_PT_snow(Panel):
    bl_space_type = "VIEW_3D"
    bl_context = "objectmode"
    bl_region_type = "UI"
    bl_label = "Snow"
    bl_category = "Real Snow"

    def draw(self, context):
        scn = context.scene
        settings = scn.snow
        layout = self.layout

        box = layout.box()
        row = box.row(align=True)
        row.prop(settings, 'live_preview', toggle=True, icon='HIDE_OFF' if settings.live_preview else 'HIDE_ON')
        sub = row.row(align=True)
        sub.enabled = settings.live_preview
        sub.operator("snow.preview_refresh", text="", icon='FILE_REFRESH')
        if settings.live_preview and _preview_state["msg"]:
            box.label(text=_preview_state["msg"], icon='INFO')

        layout.prop(settings, 'mode', expand=True)
        col = layout.column(align=True)
        col.prop(settings, 'coverage', slider=True)
        if settings.mode == 'CLUMPS':
            col.prop(settings, 'clump_size')
            col.prop(settings, 'thickness', slider=True)
        else:
            col.prop(settings, 'height')

        layout.use_property_split = True
        layout.use_property_decorate = False
        clumps = settings.mode == 'CLUMPS'

        col = layout.column(heading="Where")
        col.prop(settings, 'vertices')
        col.prop(settings, 'sky_occlusion')
        if clumps:
            # Trees and foliage
            col.prop(settings, 'use_leaf_alpha')
        else:
            col.prop(settings, 'max_slope')

        if not clumps:
            col = layout.column(heading="Drifts")
            col.prop(settings, 'wind_direction')
            col.prop(settings, 'drift_size')
            col.prop(settings, 'drift_strength', slider=True)
            col.prop(settings, 'wall_drifts')
            sub = col.column(align=True)
            sub.active = settings.wall_drifts
            sub.prop(settings, 'wall_drift_width')
            sub.prop(settings, 'wall_drift_height')

            # Roofs and ledges
            col = layout.column(heading="Roofs")
            col.prop(settings, 'icicles')
            sub = col.column(align=True)
            sub.active = settings.icicles
            sub.prop(settings, 'icicle_length')
            sub.prop(settings, 'icicle_spacing')

        col = layout.column(heading="Mesh")
        col.prop(settings, 'detail')
        col.prop(settings, 'optimize_mesh')
        if not clumps:
            col.prop(settings, 'remove_hidden')
        col.prop(settings, 'replace_existing')
        col.prop(settings, 'clump_seed')

        col = layout.column(heading="Material")
        col.prop(settings, 'snow_type')
        col.prop(settings, 'render_displacement')

        box = layout.box()
        box.label(text="Animation & Effects", icon='ANIM')
        col = box.column()
        col.prop(settings, 'animation')
        sub = col.column(align=True)
        sub.active = settings.animation != 'NONE'
        sub.prop(settings, 'anim_start')
        sub.prop(settings, 'anim_length')
        col.prop(settings, 'falling_snow')
        sub = col.column(align=True)
        sub.active = settings.falling_snow
        sub.prop(settings, 'flake_count')
        sub.prop(settings, 'flake_size')
        sub.prop(settings, 'fall_speed')

        row = layout.row(align=True)
        row.scale_y = 1.5
        row.operator("snow.create", text="Add Snow", icon="FREEZE")

        box = layout.box()
        box.label(text="Game Export", icon='EXPORT')
        col = box.column()
        col.prop(settings, 'bake_resolution')
        col.operator("snow.bake_game", text="Prepare for Game", icon='RENDER_STILL')


class SNOW_OT_Create(Operator):
    bl_idname = "snow.create"
    bl_label = "Create Snow"
    bl_description = "Create snow"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context) -> bool:
        return bool(context.selected_objects)

    def execute(self, context):
        settings = context.scene.snow
        vertices = settings.vertices

        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        clear_previews()

        # Get a list of selected objects, except non-mesh objects and snow
        input_objects = snow_targets(context)
        if not input_objects:
            self.report({'WARNING'}, "Real Snow: select at least one mesh object")
            return {'CANCELLED'}

        if settings.replace_existing:
            for obj in input_objects:
                for child in list(obj.children):
                    if child.get("real_snow"):
                        remove_snow(child)
        depsgraph = context.evaluated_depsgraph_get()
        # Walls, shelter and drifts take every selected object into account
        scene_bvh = _scene_bvh(input_objects, depsgraph)

        snow_list = []
        total_verts = 0
        alpha_cache = {}
        masks_used = set()
        skipped = []
        wm = context.window_manager
        wm.progress_begin(0, len(input_objects))
        for i, obj in enumerate(input_objects):
            wm.progress_update(i)
            snow = make_snow(context, obj, settings, settings.detail, alpha_cache, masks_used,
                             scene_bvh)
            if snow is None:
                skipped.append(obj.name)
                continue
            # Place inside collection
            coll = context.scene.collection.children.get("Snow")
            if coll is None:
                coll = bpy.data.collections.new("Snow")
                context.scene.collection.children.link(coll)
            coll.objects.link(snow)
            snow["real_snow"] = True
            # New objects get their world matrix on the next update: the
            # animation below needs the real position
            context.view_layer.update()
            # The material reads these; animations keyframe them
            snow["snow_progress"] = 1.0
            snow["snow_melt"] = 0.0
            if settings.icicles:
                add_icicles(context, snow, obj, settings)
            if settings.animation != 'NONE':
                add_snow_animation(context, snow, obj, settings)
            snow_list.append(snow)
            total_verts += len(snow.data.vertices)
        if settings.falling_snow and snow_list:
            add_falling_snow(context, input_objects, settings)

        wm.progress_end()

        if snow_list:
            bpy.ops.object.select_all(action='DESELECT')
            for s in snow_list:
                s.select_set(True)
            context.view_layer.objects.active = snow_list[0]
        if skipped:
            msg = ("Real Snow: no snow on " + ", ".join(skipped) +
                   " (no upward-facing faces" +
                   (", or no faces selected in Edit Mode" if vertices else "") +
                   ", or the surface is too small for this Height/Coverage)")
            self.report({'WARNING'}, msg)
        if not snow_list:
            return {'CANCELLED'}
        msg = "Real Snow: added snow to %d object(s), %d vertices" % (len(snow_list), total_verts)
        if masks_used:
            msg += " (leaf alpha from: %s)" % ", ".join(sorted(masks_used))
        self.report({'INFO'}, msg)
        return {'FINISHED'}


def snow_targets(context):
    """Selected meshes to put snow on.

    Snow made earlier stays selected after Add Snow: use the object it sits
    on instead, so tweaking settings and pressing Add Snow again re-does that
    object. Previews and effect objects are ignored.
    """
    out = []
    for o in context.selected_objects:
        if o.type != 'MESH' or o.get("real_snow_preview") or o.get("real_snow_fx"):
            continue
        if o.get("real_snow"):
            o = o.parent
            if o is None or o.type != 'MESH':
                continue
        if o not in out:
            out.append(o)
    return out


def remove_snow(obj):
    """Remove an object made by the add-on, with its effect children."""
    for child in list(obj.children):
        if child.get("real_snow_fx"):
            remove_snow(child)
    data = obj.data
    bpy.data.objects.remove(obj)
    if isinstance(data, bpy.types.Mesh) and data.users == 0:
        bpy.data.meshes.remove(data)


def make_snow(context, obj, settings, detail, alpha_cache=None, masks_used=None, scene_bvh=None):
    """Build the snow object for obj (not linked to the scene), or None."""
    depsgraph = context.evaluated_depsgraph_get()
    if scene_bvh is None:
        scene_bvh = _scene_bvh([obj], depsgraph)
    if settings.mode == 'CLUMPS':
        snow = build_clump_snow(context, obj, depsgraph, settings, DETAIL_RES[detail],
                                alpha_cache if alpha_cache is not None else {}, detail=detail)
        if masks_used is not None:
            masks_used.update(getattr(build_clump_snow, "masks_used", ()))
    else:
        points, _area, bvh = sample_snow_points(obj, depsgraph, settings.vertices,
                                                settings.height, settings.coverage,
                                                settings.max_slope, settings.clump_seed)
        if len(points) == 0:
            return None
        points, radii = shape_snow_points(points, settings, scene_bvh)
        if len(points) == 0:
            return None
        snow = build_snow_mesh(context, points, settings.height, DETAIL_RES[detail],
                               bvh if settings.remove_hidden else None, radii)
    if snow is None:
        return None
    if settings.optimize_mesh:
        optimize_mesh(snow.data, OPTIMIZE_RATIO[detail])
    if settings.mode != 'CLUMPS':
        add_modifiers(snow, settings.render_displacement)
    add_material(snow, settings.render_displacement, settings.snow_type)
    # Parent with object, keeping the snow where it is
    snow.parent = obj
    snow.matrix_parent_inverse = obj.matrix_world.inverted()
    return snow


def sample_snow_points(obj, depsgraph, only_selected: bool, height: float, coverage: float,
                       max_slope: float = math.radians(61.4), seed: int = 0):
    """Pick random points on the upward-facing faces of obj (world space).

    Replaces the old hair-particle emitter: same density formula, area weighted.
    """
    object_eval = obj.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(object_eval)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bpy.data.meshes.remove(mesh)
    bm.transform(obj.matrix_world)
    bm.normal_update()
    # Whole surface (world space), used later to find snow hidden inside the object
    bvh = BVHTree.FromBMesh(bm)
    up = Vector((0, 0, 1.0))
    # Snow slides off anything steeper than Max Slope
    keep = [f for f in bm.faces
            if (not only_selected or f.select)
            and up.angle(f.normal, 4.0) <= max_slope]
    keep_set = set(keep)
    tris = [tuple(l.vert.co.copy() for l in tri) for tri in bm.calc_loop_triangles() if tri[0].face in keep_set]
    bm.free()
    weights = [(b - a).cross(c - a).length * 0.5 for a, b, c in tris]
    surface_area = sum(weights)
    number = int(surface_area * BALL_DENSITY * (height ** -2) * ((coverage / 100) ** 2))
    if number <= 0 or surface_area <= 0.0:
        return np.zeros((0, 3)), surface_area, bvh
    # Vectorized with numpy (bundled with Blender)
    gen = np.random.default_rng(seed)
    a = np.array([t[0] for t in tris], dtype=np.float64)
    b = np.array([t[1] for t in tris], dtype=np.float64)
    c = np.array([t[2] for t in tris], dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    idx = gen.choice(len(w), size=number, p=w / w.sum())
    u = gen.random(number)
    v = gen.random(number)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    points = a[idx] + (b[idx] - a[idx]) * u[:, None] + (c[idx] - a[idx]) * v[:, None]
    return points, surface_area, bvh



# ---------------------------------------------------------------------------
# Clump mode: snow for foliage (pine branches, bushes, needle cards)
#
# Real snow on conifers does not form one continuous blanket: it collects in
# separate rounded pillows on top of each branch tier, with dark gaps between
# them, more on exposed outer/upper branches and little under other branches.
# Instead of covering every upward face, this mode picks spaced clump centres,
# grows each clump from the branch surface around it (so it follows the
# branch) and domes it upward.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Leaf alpha: most tree assets are built from flat "cards" (planes) with a
# needle/leaf texture whose alpha cuts out the shape. Snow must only land
# where the texture is opaque, otherwise it covers the whole invisible card
# and forms slabs floating in the air.
# ---------------------------------------------------------------------------

ALPHA_MAP_SIZE = 512
ALPHA_CUTOFF = 0.5


def _find_alpha_source(mat):
    """Return (image, channel) that drives the material's transparency, or None."""
    if mat is None or not mat.node_tree:
        return None
    nt = mat.node_tree

    def upstream_image(socket, depth=0):
        if depth > 6 or not socket.is_linked:
            return None
        link = socket.links[0]
        node = link.from_node
        if node.type == 'TEX_IMAGE' and node.image is not None:
            return node.image, (3 if link.from_socket.name == "Alpha" else 0)
        for inp in node.inputs:
            if inp.type in {'VALUE', 'RGBA', 'VECTOR'} and inp.name not in {"Vector", "UV"}:
                found = upstream_image(inp, depth + 1)
                if found:
                    return found
        return None

    # Principled BSDF Alpha, or a Mix Shader blending with Transparent BSDF
    for node in nt.nodes:
        if node.type == 'BSDF_PRINCIPLED' and "Alpha" in node.inputs:
            found = upstream_image(node.inputs["Alpha"])
            if found:
                return found
        if node.type == 'MIX_SHADER':
            if any(i.is_linked and i.links[0].from_node.type == 'BSDF_TRANSPARENT' for i in node.inputs[1:]):
                found = upstream_image(node.inputs[0])
                if found:
                    return found
    # Fallback: any image whose Alpha output is used somewhere
    for node in nt.nodes:
        if node.type == 'TEX_IMAGE' and node.image is not None and node.outputs["Alpha"].is_linked:
            return node.image, 3
    return None


class _AlphaMask:
    """Downscaled copy of an alpha texture, sampled by UV."""

    def __init__(self, image, channel):
        img = image.copy()
        try:
            n = ALPHA_MAP_SIZE
            if img.size[0] == 0 or img.size[1] == 0:
                raise ValueError("empty image")
            w = min(n, img.size[0])
            h = min(n, img.size[1])
            img.scale(w, h)
            px = [0.0] * (w * h * 4)
            img.pixels.foreach_get(px)
            self.values = px[channel::4]
            self.w, self.h = w, h
        finally:
            bpy.data.images.remove(img)
        self.name = image.name

    def opaque(self, uv) -> bool:
        x = int((uv.x % 1.0) * self.w) % self.w
        y = int((uv.y % 1.0) * self.h) % self.h
        return self.values[y * self.w + x] >= ALPHA_CUTOFF


def _alpha_masks(obj, cache):
    """Alpha mask per material slot index (None where the material is opaque)."""
    masks = []
    for slot in obj.material_slots:
        src = _find_alpha_source(slot.material)
        if src is None:
            masks.append(None)
            continue
        key = (src[0].name, src[1])
        if key not in cache:
            try:
                cache[key] = _AlphaMask(*src)
            except Exception:
                cache[key] = None
        masks.append(cache[key])
    return masks


class _Foliage:
    """World-space triangles of an object, with UVs and leaf alpha."""

    def __init__(self, obj, depsgraph, only_selected: bool, min_up: float, use_alpha: bool, cache):
        object_eval = obj.evaluated_get(depsgraph)
        mesh = bpy.data.meshes.new_from_object(object_eval)
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bpy.data.meshes.remove(mesh)
        bm.transform(obj.matrix_world)
        bm.normal_update()
        uv_layer = bm.loops.layers.uv.active
        masks = _alpha_masks(obj, cache) if (use_alpha and uv_layer is not None) else []
        self.masks_used = sorted({m.name for m in masks if m is not None})

        verts = [v.co.copy() for v in bm.verts]
        self.tri_idx = []      # vertex indices, for the BVH
        self.tri_uv = []       # (uv0, uv1, uv2) or None
        self.tri_mask = []     # _AlphaMask or None
        self.up = []           # indices of triangles snow can rest on
        for tri in bm.calc_loop_triangles():
            f = tri[0].face
            i = len(self.tri_idx)
            self.tri_idx.append(tuple(l.vert.index for l in tri))
            mask = masks[f.material_index] if f.material_index < len(masks) else None
            self.tri_mask.append(mask)
            self.tri_uv.append(tuple(l[uv_layer].uv.copy() for l in tri) if mask is not None else None)
            if only_selected and not f.select:
                continue
            # Single sided cards: a face pointing down counts as its flip side
            if abs(f.normal.z) < min_up:
                continue
            self.up.append(i)
        bm.free()
        self.verts = verts
        self.bvh = BVHTree.FromPolygons(verts, self.tri_idx)

    def tri_co(self, i):
        a, b, c = self.tri_idx[i]
        return self.verts[a], self.verts[b], self.verts[c]

    def _uv_at(self, i, u, v):
        ua, ub, uc = self.tri_uv[i]
        return ua + (ub - ua) * u + (uc - ua) * v

    def up_area(self):
        total = 0.0
        for i in self.up:
            a, b, c = self.tri_co(i)
            total += (b - a).cross(c - a).length * 0.5
        return total

    def sample(self, count, rng):
        """Random points on the upward triangles, only where the leaf is opaque."""
        if not self.up or count <= 0:
            return []
        weights = []
        for i in self.up:
            a, b, c = self.tri_co(i)
            weights.append((b - a).cross(c - a).length * 0.5)
        if sum(weights) <= 0.0:
            return []
        pts = []
        for i in rng.choices(self.up, weights=weights, k=count):
            u, v = rng.random(), rng.random()
            if u + v > 1.0:
                u, v = 1.0 - u, 1.0 - v
            mask = self.tri_mask[i]
            if mask is not None and not mask.opaque(self._uv_at(i, u, v)):
                continue
            a, b, c = self.tri_co(i)
            pts.append(a + (b - a) * u + (c - a) * v)
        return pts

    def blocked_above(self, p, max_dist):
        """True if visible foliage is above p (transparent card parts don't count)."""
        up = Vector((0.0, 0.0, 1.0))
        origin = p
        travelled = 0.0
        for _ in range(8):
            loc, _n, i, dist = self.bvh.ray_cast(origin, up, max_dist - travelled)
            if loc is None:
                return False
            mask = self.tri_mask[i]
            if mask is None:
                return True
            a, b, c = self.tri_co(i)
            bary = _barycentric(loc, a, b, c)
            if mask.opaque(self._uv_at(i, bary[1], bary[2])):
                return True
            travelled += dist + 1e-4
            origin = loc + up * 1e-4
        return False


def _barycentric(p, a, b, c):
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = v0.dot(v0), v0.dot(v1), v1.dot(v1)
    d20, d21 = v2.dot(v0), v2.dot(v1)
    den = d00 * d11 - d01 * d01
    if abs(den) < 1e-20:
        return 1.0, 0.0, 0.0
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return 1.0 - v - w, v, w


def _hash01(v, salt):
    """Stable pseudo-random number in [0, 1) for a point."""
    x = math.sin(v.x * 12.9898 + v.y * 78.233 + v.z * 37.719 + salt * 4.581) * 43758.5453
    return x - math.floor(x)


def _grid_key(p, cell):
    return (int(math.floor(p.x / cell)), int(math.floor(p.y / cell)), int(math.floor(p.z / cell)))


def _thin_points(points, spacing, rng):
    """Poisson-disk style thinning: keep points at least `spacing` apart."""
    grid = {}
    kept = []
    order = list(range(len(points)))
    rng.shuffle(order)
    for i in order:
        p = points[i]
        cx, cy, cz = _grid_key(p, spacing)
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for q in grid.get((cx + dx, cy + dy, cz + dz), ()):
                        if (q - p).length < spacing:
                            ok = False
                            break
                    if not ok: break
                if not ok: break
            if not ok: break
        if ok:
            grid.setdefault((cx, cy, cz), []).append(p)
            kept.append(p)
    return kept


# Mesh cell size factor per Detail level in Clumps mode
CLUMP_DETAIL = {'DRAFT': 2.2, 'LOW': 1.5, 'MEDIUM': 1.0, 'HIGH': 0.55, 'ULTRA': 0.4}


def build_clump_snow(context, obj, depsgraph, settings, res_mul: float, alpha_cache=None, detail=None):
    """Snow resting on foliage, as uneven pillows with gaps between them.

    The snow is a draped layer whose depth varies: a low-frequency noise
    (scaled by Clump Size) splits it into separate pillows, Coverage sets how
    much of the foliage they cover, and depth also follows how much foliage
    there is to hold it and whether a branch above shelters it.
    The layer rests on the topmost needles, so it conforms to the branch
    instead of turning into round balls.
    """
    size = settings.clump_size
    thick = settings.thickness
    rng = random.Random(settings.clump_seed)
    up = Vector((0.0, 0.0, 1.0))
    seed_off = Vector((rng.uniform(-500, 500), rng.uniform(-500, 500), rng.uniform(-500, 500)))

    foliage = _Foliage(obj, depsgraph, settings.vertices, 0.35, settings.use_leaf_alpha,
                       alpha_cache if alpha_cache is not None else {})
    build_clump_snow.masks_used = foliage.masks_used
    if not foliage.up:
        return None
    area = foliage.up_area()

    # Snow layer elements, evenly spaced over the foliage. Oversample: with
    # alpha cards many samples land on transparent texels and are dropped.
    step = size * 0.22
    oversample = 14.0 if foliage.masks_used else 6.0
    n_fine = int(min(1500000, area * oversample / (step * step)))
    fine = foliage.sample(max(n_fine, 50), rng)
    if len(fine) < 3:
        return None
    kd = KDTree(len(fine))
    for i, p in enumerate(fine):
        kd.insert(p, i)
    kd.balance()
    # Snow rests on the topmost needles: lift every sample to the top of the
    # foliage around it, then thin so each spot of the layer is counted once
    # (stacked needle layers would otherwise pile up into a huge blob)
    lifted = []
    for p in fine:
        top = max(v.z for v, _i, _d in kd.find_range(p, step * 0.75))
        lifted.append(Vector((p.x, p.y, top)))
    anchors = _thin_points(lifted, step, rng)

    # Foliage density that a "full" spot reaches (for support)
    probe = rng.sample(anchors, min(300, len(anchors)))
    counts = sorted(len(kd.find_range(q, size * 0.5)) for q in probe)
    full = max(1, counts[int(len(counts) * 0.8)])

    depth_max = size * (0.25 + 0.85 * thick)
    cover = settings.coverage / 100.0
    # Pillows: one per random cell centre (Voronoi), about 2x Clump Size
    # apart. Coverage decides how many cells get a pillow and how far it
    # spreads; the cell lookup is warped so outlines are irregular.
    cell = size * 2.0
    keep = 0.3 + 0.65 * cover
    reach = 0.3 + 0.32 * cover
    elements = []
    for p in anchors:
        q = (p + seed_off) / cell
        q = q + noise.noise_vector(q * 1.7) * 0.22
        dists, feats = noise.voronoi(q)
        f = feats[0]
        h1 = _hash01(f, 0)
        if h1 > keep:
            continue
        radius = reach * (0.75 + 0.5 * _hash01(f, 1))
        m = 1.0 - dists[0] / radius
        if m <= 0.0:
            continue
        m = math.sqrt(m * (2.0 - m))  # dome: thick middle, thin edges
        support = min(1.0, len(kd.find_range(p, size * 0.5)) / full)
        lumps = 0.8 + 0.4 * noise.noise((p + seed_off) / (size * 0.3))
        d = depth_max * m * lumps * (0.35 + 0.65 * support)
        if settings.sky_occlusion:
            if foliage.blocked_above(p + up * (size * 0.3), size * 4.0):
                d *= 0.5
        if d < step * 0.3:
            continue
        elements.append((p, d))
    if not elements:
        return None

    center = sum((e[0] for e in elements), Vector()) / len(elements)
    name = "RealSnowTmp%06d" % random.randint(0, 999999)
    ball = bpy.data.metaballs.new(name)
    ball.resolution = step * 1.1 * CLUMP_DETAIL[detail or settings.detail]
    ball.render_resolution = ball.resolution
    ball.threshold = 1.0
    # Calibrated on a flat surface: with this radius and threshold the top of
    # the layer sits about `d` above the anchors and fades to nothing at d=0
    for co, d in elements:
        el = ball.elements.new()
        el.co = co - center
        el.radius = step * 0.8 + d
        el.stiffness = 1.0

    ballobj = bpy.data.objects.new(name, ball)
    ballobj.location = center
    context.scene.collection.objects.link(ballobj)
    try:
        dg = context.evaluated_depsgraph_get()
        dg.update()
        mesh = bpy.data.meshes.new_from_object(ballobj.evaluated_get(dg))
    finally:
        bpy.data.objects.remove(ballobj)
        bpy.data.metaballs.remove(ball)
    if len(mesh.vertices) == 0:
        bpy.data.meshes.remove(mesh)
        return None
    mesh.name = "Snow"
    snow = bpy.data.objects.new("Snow", mesh)
    snow.location = center
    return snow


def _scene_bvh(objects, depsgraph):
    """One BVH of all the given objects (world space)."""
    verts, polys = [], []
    for obj in objects:
        mesh = bpy.data.meshes.new_from_object(obj.evaluated_get(depsgraph))
        mesh.transform(obj.matrix_world)
        base = len(verts)
        verts.extend(v.co.copy() for v in mesh.vertices)
        polys.extend(tuple(base + i for i in p.vertices) for p in mesh.polygons)
        bpy.data.meshes.remove(mesh)
    return BVHTree.FromPolygons(verts, polys)


def _wall_distance(p, bvh, width, rays=8):
    """Horizontal distance from p to the nearest wall (steep face facing p)."""
    origin = p + Vector((0.0, 0.0, width * 0.05))
    best = width
    for k in range(rays):
        a = 2.0 * math.pi * k / rays
        d = Vector((math.cos(a), math.sin(a), 0.0))
        loc, normal, _i, dist = bvh.ray_cast(origin, d, width)
        if loc is None or abs(normal.z) > 0.6 or normal.dot(d) > -0.2:
            continue
        best = min(best, dist)
    return best


def _per_cell(points, cell, fn):
    """Evaluate fn(cell centre x, y, point z) once per grid cell, per point."""
    keys = np.floor(points / cell).astype(np.int64)
    uniq, first, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    vals = np.empty(len(uniq))
    for j in range(len(uniq)):
        kx, ky, _kz = uniq[j]
        vals[j] = fn(Vector(((kx + 0.5) * cell, (ky + 0.5) * cell, points[first[j], 2])))
    return vals[inv.reshape(-1)]


def shape_snow_points(points, s, scene_bvh):
    """Where snow is deeper or missing (Surface mode).

    Returns the points kept and a radius factor for each (thickness scales
    with it): wind drifts, drifts piled against walls, and shelter under
    other geometry (sky occlusion). Each effect changes slowly over space,
    so it is evaluated once per small grid cell (numpy) instead of for every
    one of the (often several hundred thousand) points.
    """
    gen = np.random.default_rng(s.clump_seed + 17)
    n = len(points)
    keep = np.ones(n, dtype=bool)
    f = np.ones(n)
    up = Vector((0.0, 0.0, 1.0))
    if s.sky_occlusion:
        # Snow falls at an angle (wind, flakes drifting): the share of these
        # directions that are blocked gives a soft, gradual shelter edge
        lift = up * max(0.01, s.height * 0.3)
        dirs = [up]
        for k in range(6):
            a = 2.0 * math.pi * k / 6 + s.wind_direction
            dirs.append(Vector((math.cos(a) * 0.45, math.sin(a) * 0.45, 1.0)).normalized())

        def shelter(p):
            o = p + lift
            return sum(1 for d in dirs if scene_bvh.ray_cast(o, d, 1000.0)[0] is not None) / len(dirs)
        frac = _per_cell(points, max(s.height * 0.35, 0.01), shelter)
        # Sheltered snow is thinner, and fully sheltered spots stay bare
        f *= 1.0 - 0.85 * frac
        keep &= gen.random(n) >= frac ** 2
    if s.drift_strength > 0.0:
        wind = Vector((math.cos(s.wind_direction), math.sin(s.wind_direction), 0.0))
        side = Vector((-wind.y, wind.x, 0.0))
        size = max(s.drift_size, 1e-3)
        off = Vector(gen.uniform(-500.0, 500.0, 3))

        def drift(p):
            q = p + off
            dq = Vector((q.dot(wind) / (size * 2.5), q.dot(side) / size, q.z / size))
            return noise.noise(dq) + 0.35 * noise.noise(dq * 2.7)
        f *= 1.0 + 0.9 * s.drift_strength * _per_cell(points, max(size / 10.0, 0.01), drift)
    width = s.wall_drift_width
    if s.wall_drifts and width > 0.0:
        dist = _per_cell(points, max(width / 6.0, 0.01),
                         lambda p: _wall_distance(p, scene_bvh, width))
        t = np.clip(1.0 - dist / width, 0.0, 1.0)
        f += s.wall_drift_height * t * t
    keep &= f >= 0.3    # scoured bare by the wind
    return points[keep], f[keep]


OPTIMIZE_RATIO = {'DRAFT': 0.3, 'LOW': 0.4, 'MEDIUM': 0.5, 'HIGH': 0.65, 'ULTRA': 0.8}


def optimize_mesh(mesh, ratio):
    """Collapse-decimate: flat areas lose most vertices, curved ones keep them."""
    if ratio >= 0.999 or len(mesh.polygons) < 200:
        return
    tmp = bpy.data.objects.new("RealSnowOptimize", mesh)
    bpy.context.scene.collection.objects.link(tmp)
    try:
        mod = tmp.modifiers.new("Optimize", 'DECIMATE')
        mod.decimate_type = 'COLLAPSE'
        mod.ratio = ratio
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()
        reduced = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg))
    finally:
        bpy.data.objects.remove(tmp)
    bm = bmesh.new()
    bm.from_mesh(reduced)
    bm.to_mesh(mesh)
    bm.free()
    bpy.data.meshes.remove(reduced)
    for poly in mesh.polygons:
        poly.use_smooth = True
    mesh.update()


def add_icicles(context, snow, source, s):
    """Icicles hanging from the edges of roofs and ledges that carry snow."""
    for child in list(snow.children):
        if child.get("real_snow_fx"):
            mesh = child.data
            bpy.data.objects.remove(child)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
    depsgraph = context.evaluated_depsgraph_get()
    mesh = bpy.data.meshes.new_from_object(source.evaluated_get(depsgraph))
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bpy.data.meshes.remove(mesh)
    bm.transform(source.matrix_world)
    bm.normal_update()
    up = Vector((0.0, 0.0, 1.0))
    # Edges where snow sits on top and nothing continues outwards: the edge
    # between an upward face and a side face, or the open border of an
    # upward face (a roof plane)
    edges = []
    for e in bm.edges:
        faces = e.link_faces
        tops = [f for f in faces if up.angle(f.normal, 4.0) <= s.max_slope]
        if not tops:
            continue
        if len(faces) == 1:
            f = faces[0]
            mid = (e.verts[0].co + e.verts[1].co) * 0.5
            out = mid - f.calc_center_median()
        elif len(faces) == 2 and len(tops) == 1:
            other = faces[0] if faces[1] is tops[0] else faces[1]
            # A side or underside face (the end of a tilted roof slab
            # leans quite far): not another top face
            if other.normal.z > 0.5 or other.normal.z < -0.8:
                continue
            out = other.normal.copy()
        else:
            continue
        out.z = 0.0
        if out.length < 1e-6:
            continue
        edges.append((e.verts[0].co.copy(), e.verts[1].co.copy(), out.normalized()))
    bm.free()
    if not edges:
        return None
    # Snow above the edge is required
    from mathutils.kdtree import KDTree
    mw = snow.matrix_world
    kd = KDTree(len(snow.data.vertices))
    for i, v in enumerate(snow.data.vertices):
        kd.insert(mw @ v.co, i)
    kd.balance()
    rng = random.Random(s.clump_seed + 5)
    spacing = max(0.01, s.icicle_spacing)
    length = s.icicle_length
    out_bm = bmesh.new()
    count = 0
    for a, b, out in edges:
        span = (b - a).length
        n = int(span / spacing)
        for k in range(n):
            t = (k + rng.uniform(0.2, 0.8)) / max(n, 1)
            p = a.lerp(b, t)
            near = kd.find(p)
            if near[1] is None or near[2] > s.height * 1.5 + 0.05:
                continue
            # Icicles grow in clusters: long ones together, gaps between
            cl = 0.5 + 0.5 * noise.noise(p / max(spacing * 6.0, 0.05) + Vector((3.7, 1.1, 0.0)))
            if cl < 0.35:
                continue
            L = length * (0.3 + 0.7 * cl) * rng.uniform(0.5, 1.15)
            r = max(0.003, L * rng.uniform(0.08, 0.13))
            geom = bmesh.ops.create_cone(out_bm, cap_ends=True, segments=6,
                                         radius1=0.0005, radius2=r, depth=L)
            vs = geom["verts"]
            # Tip down, top at the edge, slightly wobbly and leaning
            tilt = Vector((rng.uniform(-0.06, 0.06), rng.uniform(-0.06, 0.06), 0.0))
            for v in vs:
                z = v.co.z - L * 0.5          # top at 0, tip at -L
                wob = 1.0 + 0.25 * math.sin(z / max(r, 1e-4) * 1.7 + k)
                v.co.x *= wob
                v.co.y *= wob
                v.co.z = z
                v.co += tilt * (-z)
                v.co += p + out * (r + 0.005) - up * 0.004
            count += 1
    if count == 0:
        out_bm.free()
        return None
    me = bpy.data.meshes.new("Icicles")
    out_bm.to_mesh(me)
    out_bm.free()
    for poly in me.polygons:
        poly.use_smooth = True
    me.materials.append(_ice_material())
    ice = bpy.data.objects.new("Icicles", me)
    ice["real_snow_fx"] = True
    for c in snow.users_collection:
        c.objects.link(ice)
    ice.parent = snow
    ice.matrix_parent_inverse = snow.matrix_world.inverted()
    return ice


def _ice_material():
    mat = bpy.data.materials.get("Ice")
    if mat is not None and mat.get("real_snow_ice"):
        return mat
    mat = bpy.data.materials.new("Ice")
    mat["real_snow_ice"] = True
    if bpy.app.version < (5, 0, 0):
        mat.use_nodes = True
    b = mat.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (0.85, 0.93, 1.0, 1.0)
    b.inputs["Roughness"].default_value = 0.08
    b.inputs["IOR"].default_value = 1.31
    b.inputs["Transmission Weight"].default_value = 1.0
    return mat


# World-space metaball element radius / resolution per unit of snow Height.
# Calibrated so the snow matches the thickness and look of the old
# particle-instanced metaballs (v1.3.x).
BALL_RADIUS = 1.5 * 0.09 * 4.0
BALL_RES = 0.09

# Metaball elements per unit area (scaled by Height^-2) and field threshold.
# Fewer elements make the snow visibly lumpy, so these match v1.4; the
# element count only affects generation time, not the final vertex count.
BALL_DENSITY = 50.0
BALL_THRESHOLD = 1.3

# Polygonization cell size multiplier for each Detail level.
# HIGH reproduces the v1.4 mesh resolution.
DETAIL_RES = {'DRAFT': 4.5, 'LOW': 3.0, 'MEDIUM': 2.0, 'HIGH': 1.0, 'ULTRA': 0.7}


def build_snow_mesh(context, points, height: float, res_mul: float = 1.0, bvh=None, radii=None):
    """Turn the sampled points into one blobby snow mesh using a metaball.

    Matches the old setup (metaball scaled 0.09, particle size = height),
    but adds the elements directly instead of instancing via particles,
    and converts with the data API instead of the Convert operator.
    """
    pts = np.asarray(points, dtype=np.float64)
    center = Vector(pts.mean(axis=0))
    rel = (pts - np.asarray(center)).tolist()
    rad = (np.asarray(radii) if radii is not None else np.ones(len(pts))).tolist()
    # Unique name so leftover "SnowBall" metaballs can never join this family
    name = "RealSnowTmp%06d" % random.randint(0, 999999)
    ball = bpy.data.metaballs.new(name)
    ball.resolution = BALL_RES * res_mul * (0.7 * height + 0.3)
    ball.render_resolution = ball.resolution
    ball.threshold = BALL_THRESHOLD
    cell = ball.resolution
    radius = BALL_RADIUS * height
    new = ball.elements.new
    for co, k in zip(rel, rad):
        el = new()
        el.co = co
        el.radius = radius * k
        el.stiffness = 0.75
    ballobj = bpy.data.objects.new(name, ball)
    ballobj.location = center
    context.scene.collection.objects.link(ballobj)
    try:
        depsgraph = context.evaluated_depsgraph_get()
        depsgraph.update()
        mesh = bpy.data.meshes.new_from_object(ballobj.evaluated_get(depsgraph))
    finally:
        bpy.data.objects.remove(ballobj)
        bpy.data.metaballs.remove(ball)
    if len(mesh.vertices) == 0:
        bpy.data.meshes.remove(mesh)
        return None
    if bvh is not None:
        remove_hidden_faces(mesh, center, bvh, cell)
        if len(mesh.vertices) == 0:
            bpy.data.meshes.remove(mesh)
            return None
    mesh.name = "Snow"
    snow = bpy.data.objects.new("Snow", mesh)
    snow.location = center
    return snow


def remove_hidden_faces(mesh, center: Vector, bvh, cell: float):
    """Delete snow faces buried inside (or under) the source object.

    Every metaball sits on the surface, so roughly half of the generated
    mesh is the underside of the blobs, which can never be seen.
    """
    bm = bmesh.new()
    bm.from_mesh(mesh)
    margin = cell * 0.5
    hidden = []
    for f in bm.faces:
        if f.normal.z > 0.2:
            continue
        p = f.calc_center_median() + center
        loc, normal, _index, _dist = bvh.find_nearest(p)
        if loc is None:
            continue
        # Behind the surface: inside the object / under a ground plane
        if (p - loc).dot(normal) < -margin:
            hidden.append(f)
    if hidden:
        bmesh.ops.delete(bm, geom=hidden, context='FACES_ONLY')
        loose = [v for v in bm.verts if not v.link_faces]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context='VERTS')
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def add_modifiers(snow, render_displacement: bool = False):
    # v1.4 used Decimate 0.5 + Subdivision (level 1 in the viewport, adaptive
    # at render time), which doubled the viewport vertex count and made Cycles
    # dice the mesh down to pixel size. Now the viewport shows the base mesh,
    # and subdivision only happens at render time when asked for.
    if not render_displacement:
        return
    subdiv = snow.modifiers.new("Subdiv", "SUBSURF")
    subdiv.levels = 0
    subdiv.render_levels = 1
    subdiv.quality = 1
    if hasattr(subdiv, "use_adaptive_subdivision"):
        # Blender 5.0+: adaptive subdivision lives on the modifier
        subdiv.use_adaptive_subdivision = True
    elif hasattr(snow, "cycles") and hasattr(snow.cycles, "use_adaptive_subdivision"):
        # Blender 4.x: Cycles object setting (experimental feature set)
        snow.cycles.use_adaptive_subdivision = True


SNOW_MAT_VERSION = 3
FRONT_SOFTNESS = 0.15


def _apply_snow_type(mat, principled, snow_type):
    col, rough, coat, sss = SNOW_TYPES.get(snow_type, SNOW_TYPES['FRESH'])
    principled.inputs["Base Color"].default_value = (*col, 1.0)
    principled.inputs["Roughness"].default_value = rough
    principled.inputs["Subsurface Weight"].default_value = sss
    if coat is not None:
        for link in list(principled.inputs["Coat Weight"].links):
            mat.node_tree.links.remove(link)
        principled.inputs["Coat Weight"].default_value = coat
    if snow_type == 'DIRTY':
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        coord = nodes.new("ShaderNodeTexCoord"); coord.location = (-1300, 900)
        dirt = nodes.new("ShaderNodeTexNoise"); dirt.location = (-1100, 900)
        dirt.inputs["Scale"].default_value = 6.0
        dirt.inputs["Detail"].default_value = 8.0
        ramp = nodes.new("ShaderNodeValToRGB"); ramp.location = (-900, 900)
        ramp.color_ramp.elements[0].position = 0.45
        ramp.color_ramp.elements[0].color = (0.82, 0.81, 0.78, 1.0)
        ramp.color_ramp.elements[1].position = 0.75
        ramp.color_ramp.elements[1].color = (0.36, 0.31, 0.25, 1.0)
        links.new(coord.outputs["Object"], dirt.inputs["Vector"])
        links.new(dirt.outputs["Fac"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], principled.inputs["Base Color"])


def _add_melt_alpha(mat, principled):
    """Alpha for the animations, read from object properties:

        x = clamp((snow_progress * (1 + w) - snow_reveal) / w)
        visible = 1 - snow_melt * x - snow_build * (1 - x)

    Snow without these properties (static, or made by older versions) has
    snow_melt = snow_build = 0 and stays fully visible.
    """
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links

    def math(op, x, y, b=None, clamp=False):
        n = nodes.new("ShaderNodeMath")
        n.operation = op
        n.use_clamp = clamp
        n.location = (x, y)
        if b is not None:
            n.inputs[1].default_value = b
        return n

    def attr(name, kind, x, y):
        n = nodes.new("ShaderNodeAttribute")
        n.attribute_type = kind
        n.attribute_name = name
        n.location = (x, y)
        return n

    w = FRONT_SOFTNESS
    progress = attr("snow_progress", 'OBJECT', -1500, 1300)
    melt = attr("snow_melt", 'OBJECT', -1500, 1100)
    build = attr("snow_build", 'OBJECT', -1500, 900)
    reveal = attr("snow_reveal", 'GEOMETRY', -1500, 1500)
    a = math('MULTIPLY', -1300, 1350, b=1.0 + w)
    links.new(progress.outputs["Fac"], a.inputs[0])
    b = math('SUBTRACT', -1150, 1400)
    links.new(a.outputs[0], b.inputs[0])
    links.new(reveal.outputs["Fac"], b.inputs[1])
    x = math('DIVIDE', -1000, 1400, b=w, clamp=True)
    links.new(b.outputs[0], x.inputs[0])
    melt_term = math('MULTIPLY', -850, 1250)
    links.new(melt.outputs["Fac"], melt_term.inputs[0])
    links.new(x.outputs[0], melt_term.inputs[1])
    one_minus_x = math('SUBTRACT', -850, 1100)
    one_minus_x.inputs[0].default_value = 1.0
    links.new(x.outputs[0], one_minus_x.inputs[1])
    build_term = math('MULTIPLY', -700, 1050)
    links.new(build.outputs["Fac"], build_term.inputs[0])
    links.new(one_minus_x.outputs[0], build_term.inputs[1])
    vis_a = math('SUBTRACT', -700, 1250)
    vis_a.inputs[0].default_value = 1.0
    links.new(melt_term.outputs[0], vis_a.inputs[1])
    vis = math('SUBTRACT', -550, 1200, clamp=True)
    links.new(vis_a.outputs[0], vis.inputs[0])
    links.new(build_term.outputs[0], vis.inputs[1])
    # Melting snow breaks up into patches rather than fading uniformly
    patch = nodes.new("ShaderNodeTexNoise")
    patch.location = (-850, 850)
    patch.inputs["Scale"].default_value = 30.0
    patch.inputs["Detail"].default_value = 3.0
    coord = nodes.new("ShaderNodeTexCoord")
    coord.location = (-1050, 850)
    links.new(coord.outputs["Object"], patch.inputs["Vector"])
    norm = nodes.new("ShaderNodeMapRange")
    norm.location = (-650, 900)
    norm.inputs["From Min"].default_value = 0.3
    norm.inputs["From Max"].default_value = 0.7
    norm.inputs["To Min"].default_value = 0.01
    norm.inputs["To Max"].default_value = 0.99
    links.new(patch.outputs["Fac"], norm.inputs["Value"])
    alpha = math('GREATER_THAN', -400, 1100)
    links.new(vis.outputs[0], alpha.inputs[0])
    links.new(norm.outputs["Result"], alpha.inputs[1])
    links.new(alpha.outputs[0], principled.inputs["Alpha"])
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = 'DITHERED'
    elif hasattr(mat, "blend_method"):
        mat.blend_method = 'HASHED'


SNOW_TYPES = {
    # base colour, roughness, coat weight, subsurface weight
    'FRESH': ((0.904, 0.904, 0.904), 0.10, None, 1.0),
    'PACKED': ((0.80, 0.81, 0.83), 0.45, 0.0, 0.8),
    'WET': ((0.70, 0.74, 0.80), 0.04, 1.0, 0.6),
    'DIRTY': ((0.78, 0.76, 0.72), 0.35, 0.0, 0.7),
}


def add_material(obj: bpy.types.Object, render_displacement: bool = False, snow_type='FRESH'):
    # Reuse the snow material instead of creating a new copy for every object
    mat_name = ("Snow" if snow_type == 'FRESH' else "Snow " + snow_type.title()) + \
        (" Displaced" if render_displacement else "")
    mat = bpy.data.materials.get(mat_name)
    if mat is not None and mat.get("real_snow") == SNOW_MAT_VERSION:
        obj.data.materials.append(mat)
        return
    if mat is None:
        mat = bpy.data.materials.new(mat_name)
    mat["real_snow"] = SNOW_MAT_VERSION
    if bpy.app.version < (5, 0, 0):
        # Always enabled (and deprecated) in Blender 5.0+
        mat.use_nodes = True
    nodes = mat.node_tree.nodes
    # Delete all nodes
    nodes.clear()
    # Add nodes
    output = nodes.new('ShaderNodeOutputMaterial')
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    vec_math = nodes.new('ShaderNodeVectorMath')
    com_xyz = nodes.new('ShaderNodeCombineXYZ')
    dis = nodes.new('ShaderNodeDisplacement')
    mul1 = nodes.new('ShaderNodeMath')
    add1 = nodes.new('ShaderNodeMath')
    add2 = nodes.new('ShaderNodeMath')
    mul2 = nodes.new('ShaderNodeMath')
    mul3 = nodes.new('ShaderNodeMath')
    range1 = nodes.new('ShaderNodeMapRange')
    range2 = nodes.new('ShaderNodeMapRange')
    range3 = nodes.new('ShaderNodeMapRange')
    vor = nodes.new('ShaderNodeTexVoronoi')
    noise1 = nodes.new('ShaderNodeTexNoise')
    noise2 = nodes.new('ShaderNodeTexNoise')
    noise3 = nodes.new('ShaderNodeTexNoise')
    mapping = nodes.new('ShaderNodeMapping')
    coord = nodes.new('ShaderNodeTexCoord')
    # Change location
    output.location = (100, 0)
    principled.location = (-200, 600)
    vec_math.location = (-400, 400)
    com_xyz.location = (-600, 400)
    dis.location = (-200, -100)
    mul1.location = (-400, -100)
    add1.location = (-600, -100)
    add2.location = (-800, -100)
    mul2.location = (-1000, -100)
    mul3.location = (-1000, -300)
    range1.location = (-400, 200)
    range2.location = (-1200, -300)
    range3.location = (-800, -300)
    vor.location = (-1500, 200)
    noise1.location = (-1500, 0)
    noise2.location = (-1500, -250)
    noise3.location = (-1500, -500)
    mapping.location = (-1700, 0)
    coord.location = (-1900, 0)
    # Change node parameters
    principled.distribution = "MULTI_GGX"
    principled.subsurface_method = "RANDOM_WALK_SKIN"
    principled.inputs["Base Color"].default_value[0] = 0.904
    principled.inputs["Base Color"].default_value[1] = 0.904
    principled.inputs["Base Color"].default_value[2] = 0.904
    principled.inputs["Subsurface Weight"].default_value = 1
    principled.inputs["Subsurface Scale"].default_value = 1
    principled.inputs["Subsurface Radius"].default_value[0] = 0.36
    principled.inputs["Subsurface Radius"].default_value[1] = 0.46
    principled.inputs["Subsurface Radius"].default_value[2] = 0.6
    principled.inputs["Specular IOR Level"].default_value = 0.224
    principled.inputs["Roughness"].default_value = 0.1
    principled.inputs["Coat Roughness"].default_value = 0.1
    principled.inputs["Coat IOR"].default_value = 1.2
    vec_math.operation = "MULTIPLY"
    vec_math.inputs[1].default_value[0] = 0.5
    vec_math.inputs[1].default_value[1] = 0.5
    vec_math.inputs[1].default_value[2] = 0.5
    com_xyz.inputs[0].default_value = 0.36
    com_xyz.inputs[1].default_value = 0.46
    com_xyz.inputs[2].default_value = 0.6
    dis.inputs["Midlevel"].default_value = 0.1
    dis.inputs["Scale"].default_value = 0.3
    mul1.operation = "MULTIPLY"
    mul1.inputs[1].default_value = 0.1
    mul2.operation = "MULTIPLY"
    mul2.inputs[1].default_value = 0.6
    mul3.operation = "MULTIPLY"
    mul3.inputs[1].default_value = 0.4
    range1.inputs[1].default_value = 0.525
    range1.inputs[2].default_value = 0.58
    range2.inputs[1].default_value = 0.069
    range2.inputs[2].default_value = 0.757
    range3.inputs[1].default_value = 0.069
    range3.inputs[2].default_value = 0.757
    vor.feature = "N_SPHERE_RADIUS"
    vor.inputs["Scale"].default_value = 30
    noise1.inputs["Scale"].default_value = 12
    noise2.inputs["Scale"].default_value = 2
    noise2.inputs["Detail"].default_value = 4
    noise3.inputs["Scale"].default_value = 1
    noise3.inputs["Detail"].default_value = 4
    mapping.inputs["Scale"].default_value[0] = 12
    mapping.inputs["Scale"].default_value[1] = 12
    mapping.inputs["Scale"].default_value[2] = 12
    # Link nodes
    link = mat.node_tree.links
    link.new(principled.outputs["BSDF"], output.inputs["Surface"])
    link.new(vec_math.outputs["Vector"], principled.inputs["Subsurface Radius"])
    link.new(com_xyz.outputs[0], vec_math.inputs[0])
    link.new(dis.outputs["Displacement"], output.inputs["Displacement"])
    link.new(mul1.outputs[0], dis.inputs["Height"])
    link.new(add1.outputs[0], mul1.inputs[0])
    link.new(add2.outputs[0], add1.inputs[0])
    link.new(mul2.outputs[0], add2.inputs[0])
    link.new(mul3.outputs[0], add2.inputs[1])
    link.new(range1.outputs[0], principled.inputs["Coat Weight"])
    link.new(range2.outputs[0], mul3.inputs[0])
    link.new(range3.outputs[0], add1.inputs[1])
    link.new(vor.outputs["Radius"], range1.inputs[0])
    link.new(noise1.outputs[0], mul2.inputs[0])
    link.new(noise2.outputs[0], range2.inputs[0])
    link.new(noise3.outputs[0], range3.inputs[0])
    link.new(mapping.outputs["Vector"], vor.inputs["Vector"])
    link.new(mapping.outputs["Vector"], noise1.inputs["Vector"])
    link.new(mapping.outputs["Vector"], noise2.inputs["Vector"])
    link.new(mapping.outputs["Vector"], noise3.inputs["Vector"])
    link.new(coord.outputs["Object"], mapping.inputs["Vector"])
    _apply_snow_type(mat, principled, snow_type)
    _add_melt_alpha(mat, principled)
    # Set displacement and add material
    # True displacement needs the render-time subdivision; bump otherwise
    mat.displacement_method = "BOTH" if render_displacement else "BUMP"
    obj.data.materials.append(mat)


# ---------------------------------------------------------------------------
# Animation & effects
# ---------------------------------------------------------------------------

def _world_bvh(obj, depsgraph):
    mesh = bpy.data.meshes.new_from_object(obj.evaluated_get(depsgraph))
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bpy.data.meshes.remove(mesh)
    bm.transform(obj.matrix_world)
    bvh = BVHTree.FromBMesh(bm)
    bm.free()
    return bvh


def add_snow_animation(context, snow, source, s):
    """Build Up (snowfall) or Melt, keyframed over the frame range.

    Build Up: the whole layer grows gradually out of the surface while the
    material fills in from scattered patches.
    Melt: thin and steep snow goes first, the thick pillows last; the
    material breaks up into patches as it goes.
    The geometry is the smooth Geometry Nodes animation; Prepare for Game
    turns it into morph targets.
    """
    start = s.anim_start
    length = max(1, s.anim_length)
    melt = s.animation == 'MELT'
    depsgraph = context.evaluated_depsgraph_get()
    bvh = _world_bvh(source, depsgraph)
    mesh = snow.data
    mw = snow.matrix_world
    inv = mw.inverted()

    # Thickness (distance to the surface) and where each vertex sinks to:
    # the nearest point of the surface, pushed just inside. Every vertex
    # travels at most its own snow depth, so nothing is dragged down a side.
    thick, flat = [], []
    for v in mesh.vertices:
        p = mw @ v.co
        near, nrm, _i, dist = bvh.find_nearest(p)
        if near is None:
            thick.append(0.0)
            flat.append(p)
            continue
        thick.append(dist)
        flat.append((near, nrm))
    tmax = max(thick) or 1.0
    sink = tmax * 0.15
    flat = [inv @ (f[0] - f[1] * sink) if isinstance(f, tuple) else inv @ f for f in flat]

    scale = max(tmax * 6.0, 0.05)
    normals = [(mw.to_3x3() @ v.normal).normalized() for v in mesh.vertices]
    times = []
    for v, t, n in zip(mesh.vertices, thick, normals):
        p = mw @ v.co
        jitter = 0.5 + 0.5 * noise.noise(p / scale)
        tn = t / tmax
        if melt:
            # thin, steep and patchy spots melt first
            tau = 0.55 * tn + 0.3 * jitter + 0.15 * max(0.0, n.z)
        else:
            # snowfall settles everywhere, a little patchy at first
            tau = 0.3 * jitter
        times.append(min(1.0, max(0.0, tau)))
    if melt and len(times) > 1:
        # Rank instead of raw value: the melted fraction then grows steadily
        # over the frame range instead of bunching up at the end
        order = sorted(range(len(times)), key=times.__getitem__)
        ranked = [0.0] * len(times)
        for r, i in enumerate(order):
            ranked[i] = r / (len(times) - 1)
        times = ranked
    end = start + length
    # Geometry: smooth Geometry Nodes animation, every vertex on its own
    # schedule. Build Up uses a very soft front, so the whole layer grows
    # gradually; Melt a sharper one, so thin spots go well before thick ones.
    add_anim_modifier(snow, times, flat, softness=0.35 if melt else 0.7,
                      invert=melt, start=start, length=length)

    snow["snow_melt"] = 1.0 if melt else 0.0
    snow["snow_build"] = 0.0 if melt else 1.0
    # Material: Build Up patches appear over the first third; Melt breaks up
    # over the whole range
    snow["snow_progress"] = 0.0
    snow.keyframe_insert('["snow_progress"]', frame=start)
    snow["snow_progress"] = 1.0
    snow.keyframe_insert('["snow_progress"]', frame=end if melt else start + max(1, length // 3))
    _linear_keys(snow)


def _flake_object():
    name = "Snow Flake"
    obj = bpy.data.objects.get(name)
    if obj is None or obj.type != 'MESH':
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=1, radius=1.0)
        bmesh.ops.scale(bm, vec=(1.0, 1.0, 0.35), verts=bm.verts)
        mesh = bpy.data.meshes.new(name)
        bm.to_mesh(mesh)
        bm.free()
        obj = bpy.data.objects.new(name, mesh)
        obj["real_snow_fx"] = True
        _fx_collection().objects.link(obj)
        obj.location = (0.0, 0.0, -1000.0)
        obj.hide_set(True)
        mat = bpy.data.materials.new("Snow Flake")
        if bpy.app.version < (5, 0, 0):
            mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = (0.95, 0.97, 1.0, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.4
        bsdf.inputs["Subsurface Weight"].default_value = 0.5
        mesh.materials.append(mat)
    return obj


def _fx_collection():
    coll = bpy.data.collections.get("Snow FX")
    if coll is None:
        coll = bpy.data.collections.new("Snow FX")
        bpy.context.scene.collection.children.link(coll)
    return coll


def add_falling_snow(context, targets, s):
    """Snowflakes falling onto the selection from a hidden emitter above it."""
    old = bpy.data.objects.get("Snowfall")
    if old is not None:
        bpy.data.objects.remove(old)
    pts = [o.matrix_world @ Vector(c) for o in targets for c in o.bound_box]
    x0, x1 = min(p.x for p in pts), max(p.x for p in pts)
    y0, y1 = min(p.y for p in pts), max(p.y for p in pts)
    z0, z1 = min(p.z for p in pts), max(p.z for p in pts)
    margin = 0.25 * max(x1 - x0, y1 - y0, 1.0)
    height = max(2.0, (z1 - z0) * 0.5)
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
    mesh = bpy.data.meshes.new("Snowfall")
    bm.to_mesh(mesh)
    bm.free()
    em = bpy.data.objects.new("Snowfall", mesh)
    em["real_snow_fx"] = True
    _fx_collection().objects.link(em)
    em.location = ((x0 + x1) / 2, (y0 + y1) / 2, z1 + height)
    em.scale = (x1 - x0 + 2 * margin, y1 - y0 + 2 * margin, 1.0)
    em.display_type = 'WIRE'
    em.hide_render = False

    mod = em.modifiers.new("Snowfall", 'PARTICLE_SYSTEM')
    psys = mod.particle_system
    psys.seed = s.clump_seed
    st = psys.settings
    st.name = "Snowfall"
    scene = context.scene
    fps = scene.render.fps / max(scene.render.fps_base, 1e-6)
    fall = (z1 + height) - z0
    speed = max(0.05, s.fall_speed)
    if s.animation != 'NONE':
        st.frame_start = s.anim_start - fall / speed * fps
        st.frame_end = s.anim_start + max(1, s.anim_length)
    else:
        st.frame_start = scene.frame_start - fall / speed * fps
        st.frame_end = scene.frame_end
    st.count = s.flake_count
    # Live until they reach the lowest selected object, then vanish
    st.lifetime = max(2, int(fall / speed * fps))
    st.lifetime_random = 0.15
    st.emit_from = 'FACE'
    st.use_emit_random = True
    st.physics_type = 'NEWTON'
    st.normal_factor = 0.0
    st.object_align_factor = (0.0, 0.0, -speed)
    st.factor_random = 0.1 * speed
    st.brownian_factor = 0.3 * speed
    st.drag_factor = 0.0
    st.effector_weights.all = 0.0     # ignore scene force fields
    st.effector_weights.gravity = 0.0  # constant terminal velocity
    st.render_type = 'OBJECT'
    st.instance_object = _flake_object()
    st.particle_size = s.flake_size
    st.size_random = 0.6
    st.use_rotations = True
    st.rotation_mode = 'NONE'
    st.phase_factor_random = 2.0
    st.use_dynamic_rotation = True
    st.angular_velocity_mode = 'RAND'
    st.angular_velocity_factor = 2.0
    st.display_method = 'DOT'
    st.display_percentage = 30
    em.show_instancer_for_render = False
    em.show_instancer_for_viewport = False
    return em


GAME_SMART_UVS = True    # snow on trees overlaps from above: unwrap properly

# ---------------------------------------------------------------------------
# Smooth animation (Geometry Nodes) and game export
# ---------------------------------------------------------------------------
#
# The animation is a Geometry Nodes modifier: every vertex has its own reveal
# time (attribute "snow_reveal") and an offset that presses it into the surface
# (attribute "snow_flat_offset"). One keyframed "Progress" input moves a soft
# front across the mesh, so each vertex moves smoothly on its own schedule.
# Game engines cannot read Geometry Nodes, so Prepare for Game samples the
# animation into morph targets (shape keys) that blend from one sample to
# the next.

ANIM_MOD = "Snow Animation"
ANIM_GROUP = "Real Snow Animation"
ANIM_GROUP_VERSION = 1
GAME_SAMPLES = 10


def _anim_node_group():
    tree = bpy.data.node_groups.get(ANIM_GROUP)
    if tree is not None and tree.get("version") == ANIM_GROUP_VERSION:
        return tree
    if tree is None:
        tree = bpy.data.node_groups.new(ANIM_GROUP, 'GeometryNodeTree')
    else:
        tree.nodes.clear()
        tree.interface.clear()
    tree["version"] = ANIM_GROUP_VERSION
    iface = tree.interface
    iface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    iface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    s_prog = iface.new_socket(name="Progress", in_out='INPUT', socket_type='NodeSocketFloat')
    s_prog.default_value = 1.0
    s_soft = iface.new_socket(name="Softness", in_out='INPUT', socket_type='NodeSocketFloat')
    s_soft.default_value = 0.25
    s_inv = iface.new_socket(name="Invert", in_out='INPUT', socket_type='NodeSocketFloat')
    s_inv.default_value = 0.0
    N, L = tree.nodes, tree.links
    gi = N.new("NodeGroupInput"); gi.location = (-900, 0)
    go = N.new("NodeGroupOutput"); go.location = (500, 0)
    reveal = N.new("GeometryNodeInputNamedAttribute"); reveal.location = (-900, -250)
    reveal.data_type = 'FLOAT'
    reveal.inputs["Name"].default_value = "snow_reveal"
    offset = N.new("GeometryNodeInputNamedAttribute"); offset.location = (-300, -400)
    offset.data_type = 'FLOAT_VECTOR'
    offset.inputs["Name"].default_value = "snow_flat_offset"
    # x = smoothstep(0, w, progress * (1 + w) - reveal)
    one_w = N.new("ShaderNodeMath"); one_w.operation = 'ADD'; one_w.location = (-700, 150)
    one_w.inputs[1].default_value = 1.0
    L.new(gi.outputs["Softness"], one_w.inputs[0])
    scaled = N.new("ShaderNodeMath"); scaled.operation = 'MULTIPLY'; scaled.location = (-550, 100)
    L.new(gi.outputs["Progress"], scaled.inputs[0])
    L.new(one_w.outputs[0], scaled.inputs[1])
    rel = N.new("ShaderNodeMath"); rel.operation = 'SUBTRACT'; rel.location = (-400, 0)
    L.new(scaled.outputs[0], rel.inputs[0])
    L.new(reveal.outputs["Attribute"], rel.inputs[1])
    x = N.new("ShaderNodeMapRange"); x.location = (-250, 0)
    x.interpolation_type = 'SMOOTHSTEP'
    x.clamp = True
    x.inputs["From Min"].default_value = 0.0
    L.new(gi.outputs["Softness"], x.inputs["From Max"])
    L.new(rel.outputs[0], x.inputs["Value"])
    # Build: pressed until the front passes (1 - x). Invert: pressed after it (x).
    one_minus = N.new("ShaderNodeMath"); one_minus.operation = 'SUBTRACT'; one_minus.location = (-50, 100)
    one_minus.inputs[0].default_value = 1.0
    L.new(x.outputs["Result"], one_minus.inputs[1])
    mix = N.new("ShaderNodeMix"); mix.location = (100, 0)
    mix.data_type = 'FLOAT'
    L.new(gi.outputs["Invert"], mix.inputs["Factor"])
    L.new(one_minus.outputs[0], mix.inputs["A"])
    L.new(x.outputs["Result"], mix.inputs["B"])
    scale = N.new("ShaderNodeVectorMath"); scale.operation = 'SCALE'; scale.location = (250, -250)
    L.new(offset.outputs["Attribute"], scale.inputs[0])
    L.new(mix.outputs["Result"], scale.inputs["Scale"])
    setp = N.new("GeometryNodeSetPosition"); setp.location = (300, 0)
    L.new(gi.outputs["Geometry"], setp.inputs["Geometry"])
    L.new(scale.outputs["Vector"], setp.inputs["Offset"])
    L.new(setp.outputs["Geometry"], go.inputs["Geometry"])
    return tree


def _socket_id(tree, name):
    for item in tree.interface.items_tree:
        if getattr(item, "in_out", None) == 'INPUT' and item.name == name:
            return item.identifier
    raise KeyError(name)


def add_anim_modifier(obj, times, flat_local, softness, invert, start, length):
    """Add the smooth animation to obj.

    times: per-vertex reveal time 0..1; flat_local: per-vertex position (object
    space) when pressed into the surface; invert: True to remove the layer as
    the front passes (blow away / melt), False to build it up.
    """
    mesh = obj.data
    reveal = mesh.attributes.get("snow_reveal") or mesh.attributes.new("snow_reveal", 'FLOAT', 'POINT')
    reveal.data.foreach_set("value", times)
    off = mesh.attributes.get("snow_flat_offset") or \
        mesh.attributes.new("snow_flat_offset", 'FLOAT_VECTOR', 'POINT')
    vals = []
    for v, f in zip(mesh.vertices, flat_local):
        d = f - v.co
        vals.extend((d.x, d.y, d.z))
    off.data.foreach_set("vector", vals)
    old = obj.modifiers.get(ANIM_MOD)
    if old is not None:
        obj.modifiers.remove(old)
    mod = obj.modifiers.new(ANIM_MOD, 'NODES')
    tree = _anim_node_group()
    mod.node_group = tree
    # First in the stack, so particles emit from the animated surface
    idx = list(obj.modifiers).index(mod)
    if idx > 0:
        obj.modifiers.move(idx, 0)
    pid = _socket_id(tree, "Progress")
    mod[_socket_id(tree, "Softness")] = softness
    mod[_socket_id(tree, "Invert")] = 1.0 if invert else 0.0
    path = 'modifiers["%s"]["%s"]' % (ANIM_MOD, pid)
    mod[pid] = 0.0
    obj.keyframe_insert(data_path=path, frame=start)
    mod[pid] = 1.0
    obj.keyframe_insert(data_path=path, frame=start + length)
    obj["snow_anim_range"] = (float(start), float(start + length))
    obj["snow_anim_invert"] = 1.0 if invert else 0.0
    _linear_keys(obj)


def _fcurves(action):
    """F-curves of an action (Blender 5 keeps them in layered channelbags)."""
    try:
        if action.fcurves is not None:
            return [(None, fc) for fc in action.fcurves]
    except AttributeError:
        pass
    curves = []
    for layer in getattr(action, "layers", []):
        for strip in layer.strips:
            for bag in getattr(strip, "channelbags", []):
                curves.extend((bag, fc) for fc in bag.fcurves)
    return curves


def _linear_keys(id_data):
    if id_data.animation_data and id_data.animation_data.action:
        for _bag, fc in _fcurves(id_data.animation_data.action):
            for kp in fc.keyframe_points:
                kp.interpolation = 'LINEAR'


def _remove_fcurves(id_data, prefix):
    ad = id_data.animation_data
    if not ad or not ad.action:
        return
    for bag, fc in _fcurves(ad.action):
        if fc.data_path.startswith(prefix):
            (bag.fcurves if bag is not None else ad.action.fcurves).remove(fc)


def bake_anim_to_shape_keys(context, obj, samples=GAME_SAMPLES):
    """Replace the Geometry Nodes animation by morph targets (game engines).

    Samples the animation at `samples` + 1 progress values. The basis is the
    full layer; each shape key moves the mesh from one sample to the next and
    is keyed over its own slice of the frame range, so playback reproduces the
    animation (linearly between samples) and engines import it as morph
    target animation.
    """
    mod = obj.modifiers.get(ANIM_MOD)
    if mod is None or mod.node_group is None:
        return False
    pid = _socket_id(mod.node_group, "Progress")
    start, end = obj.get("snow_anim_range", (1.0, 121.0))
    invert = obj.get("snow_anim_invert", 0.0) > 0.5
    path = 'modifiers["%s"]["%s"]' % (ANIM_MOD, pid)
    _remove_fcurves(obj, path)
    # Only the animation modifier: other modifiers stay live on top
    others = [(m, m.show_viewport) for m in obj.modifiers if m != mod]
    for m, _vis in others:
        m.show_viewport = False
    shapes = []
    try:
        for k in range(samples + 1):
            mod[pid] = k / samples
            obj.update_tag()
            dg = context.evaluated_depsgraph_get()
            dg.update()
            ev = obj.evaluated_get(dg)
            me = ev.to_mesh()
            co = [0.0] * (len(me.vertices) * 3)
            me.vertices.foreach_get("co", co)
            ev.to_mesh_clear()
            shapes.append(co)
    finally:
        for m, vis in others:
            m.show_viewport = vis
    nverts = len(obj.data.vertices)
    if any(len(c) != nverts * 3 for c in shapes):
        return False
    obj.modifiers.remove(mod)
    if obj.data.shape_keys:
        obj.shape_key_clear()
    # Basis = the full layer (start of a removal, end of a build up)
    order = list(range(samples + 1)) if invert else list(range(samples, -1, -1))
    obj.data.vertices.foreach_set("co", shapes[order[0]])
    obj.data.update()
    obj.shape_key_add(name="Basis", from_mix=False)
    span = end - start
    for j in range(1, len(order)):
        a, b = shapes[order[j - 1]], shapes[order[j]]
        key = obj.shape_key_add(name="%s %02d" % ("Snow", j), from_mix=False)
        base = shapes[order[0]]
        key.data.foreach_set("co", [base[i] + (b[i] - a[i]) for i in range(len(base))])
        if invert:
            p0, p1 = (j - 1) / samples, j / samples      # key j: 0 -> 1
            v0, v1 = 0.0, 1.0
        else:
            p0, p1 = (samples - j) / samples, (samples - j + 1) / samples  # 1 -> 0
            v0, v1 = 1.0, 0.0
        key.value = v0
        key.keyframe_insert("value", frame=start + p0 * span)
        key.value = v1
        key.keyframe_insert("value", frame=start + p1 * span)
    _linear_keys(obj.data.shape_keys)
    return True


def _top_uvs(obj, uv):
    """Top-down UV projection fitted to the footprint (for mostly flat layers)."""
    mesh = obj.data
    xs = [v.co.x for v in mesh.vertices]
    ys = [v.co.y for v in mesh.vertices]
    x0, y0 = min(xs), min(ys)
    size = max(max(xs) - x0, max(ys) - y0, 1e-6)
    pad = 0.01
    k = (1.0 - 2.0 * pad) / size
    for loop in mesh.loops:
        co = mesh.vertices[loop.vertex_index].co
        uv.data[loop.index].uv = (pad + (co.x - x0) * k, pad + (co.y - y0) * k)


def _make_uvs(context, obj, smart):
    mesh = obj.data
    uv = mesh.uv_layers.get("GameUV") or mesh.uv_layers.new(name="GameUV")
    mesh.uv_layers.active = uv
    if smart:
        # Overlapping layers (snow on several branch tiers) need real unwrapping
        view_layer = context.view_layer
        prev_active = view_layer.objects.active
        prev_sel = list(context.selected_objects)
        try:
            for o in prev_sel:
                o.select_set(False)
            obj.select_set(True)
            view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=0.01)
            bpy.ops.object.mode_set(mode='OBJECT')
            return "GameUV"
        except Exception:
            if obj.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
        finally:
            obj.select_set(False)
            for o in prev_sel:
                o.select_set(True)
            view_layer.objects.active = prev_active
    _top_uvs(obj, uv)
    return "GameUV"


def _bake_pass(context, obj, mat, image, bake_type, emit_input=None):
    """Bake one pass of `mat` on `obj` into `image`.

    emit_input: a BSDF input socket to bake through an emission shader (its
    link, or its plain value when nothing is connected).
    """
    nt = mat.node_tree
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = image
    nt.nodes.active = tex
    bsdf = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
    out = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output) \
        if any(n.type == 'OUTPUT_MATERIAL' and n.is_active_output for n in nt.nodes) \
        else next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
    saved_surface = out.inputs["Surface"].links[0].from_socket if out.inputs["Surface"].links else None
    alpha_link = bsdf.inputs["Alpha"].links[0].from_socket if bsdf.inputs["Alpha"].links else None
    alpha_value = bsdf.inputs["Alpha"].default_value
    # What to bake through emission: read it before the alpha link is cut
    emit_src = None
    emit_val = None
    if emit_input is not None:
        if emit_input.links:
            emit_src = emit_input.links[0].from_socket
        else:
            emit_val = emit_input.default_value
    temp = []
    try:
        # Opaque while baking colour/normal/roughness
        if alpha_link is not None:
            nt.links.remove(bsdf.inputs["Alpha"].links[0])
        bsdf.inputs["Alpha"].default_value = 1.0
        if emit_input is not None:
            emit = nt.nodes.new("ShaderNodeEmission")
            temp.append(emit)
            if emit_src is not None:
                nt.links.new(emit_src, emit.inputs["Color"])
            else:
                val = emit_val
                if hasattr(val, "__len__"):
                    emit.inputs["Color"].default_value = tuple(val)[:3] + (1.0,)
                else:
                    emit.inputs["Color"].default_value = (val, val, val, 1.0)
            nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
        bpy.ops.object.bake(type=bake_type, margin=8, use_clear=True, normal_space='TANGENT')
    finally:
        for n in temp:
            nt.nodes.remove(n)
        if saved_surface is not None:
            nt.links.new(saved_surface, out.inputs["Surface"])
        if alpha_link is not None:
            nt.links.new(alpha_link, bsdf.inputs["Alpha"])
        bsdf.inputs["Alpha"].default_value = alpha_value
        nt.nodes.remove(tex)


FULL_STATE_PROPS = {"snow_progress": 1.0, "snow_blow": 0.0, "snow_melt": 0.0, "snow_build": 0.0}


def bake_for_game(context, obj, res, smart_uvs):
    """Bake obj's material to images, turn its animation into morph targets
    and swap in a material that every exporter understands."""
    src_mat = obj.active_material
    if src_mat is None or not src_mat.node_tree:
        return False
    if obj.modifiers.get(ANIM_MOD) is not None:
        bake_anim_to_shape_keys(context, obj)
    uv_name = _make_uvs(context, obj, smart_uvs)
    name = obj.name.replace(" ", "_").replace(".", "_")

    def img(suffix, alpha=False, data=False):
        full = "%s_%s" % (name, suffix)
        old = bpy.data.images.get(full)
        if old is not None:
            bpy.data.images.remove(old)
        im = bpy.data.images.new(full, res, res, alpha=alpha, float_buffer=False)
        if data:
            im.colorspace_settings.name = 'Non-Color'
        return im
    color = img("BaseColor", alpha=True)
    alpha_tmp = img("AlphaTmp", data=True)
    normal = img("Normal", data=True)
    rough = img("Roughness", data=True)

    mat = src_mat.copy()   # bake on a private copy (the material is shared)
    obj.active_material = mat
    bsdf = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    has_alpha = bool(bsdf.inputs["Alpha"].links)

    # Bake the full, static layer whatever frame the animation is on
    saved = {k: obj.get(k) for k in FULL_STATE_PROPS}
    for k, v in FULL_STATE_PROPS.items():
        obj[k] = v
    prev_only = (obj.show_only_shape_key, obj.active_shape_key_index)
    obj.show_only_shape_key = True
    obj.active_shape_key_index = 0
    scene = context.scene
    prev_engine = scene.render.engine
    prev_samples = scene.cycles.samples
    view_layer = context.view_layer
    prev_sel = list(context.selected_objects)
    prev_active = view_layer.objects.active
    try:
        scene.render.engine = 'CYCLES'
        scene.cycles.samples = 16
        for o in prev_sel:
            o.select_set(False)
        obj.select_set(True)
        view_layer.objects.active = obj
        _bake_pass(context, obj, mat, color, 'EMIT', emit_input=bsdf.inputs["Base Color"])
        if has_alpha:
            _bake_pass(context, obj, mat, alpha_tmp, 'EMIT', emit_input=bsdf.inputs["Alpha"])
        _bake_pass(context, obj, mat, normal, 'NORMAL')
        _bake_pass(context, obj, mat, rough, 'ROUGHNESS')
    finally:
        scene.render.engine = prev_engine
        scene.cycles.samples = prev_samples
        for k, v in saved.items():
            if v is None:
                obj.pop(k, None)
            else:
                obj[k] = v
        obj.show_only_shape_key, obj.active_shape_key_index = prev_only
        obj.select_set(False)
        for o in prev_sel:
            o.select_set(True)
        view_layer.objects.active = prev_active
        bpy.data.materials.remove(mat)

    # Alpha into the colour image's alpha channel; opaque if it is all solid
    n = res * res * 4
    cpx = [0.0] * n
    color.pixels.foreach_get(cpx)
    cutout = False
    if has_alpha:
        apx = [0.0] * n
        alpha_tmp.pixels.foreach_get(apx)
        alpha = apx[0::4]
        # Only texels the mesh covers count (the space between UV islands
        # bakes as alpha 0): cut out only if real surface is transparent
        covered = [i for i in range(res * res)
                   if cpx[i * 4] + cpx[i * 4 + 1] + cpx[i * 4 + 2] > 0.002]
        holes = sum(1 for i in covered if alpha[i] < 0.5)
        cutout = bool(covered) and holes > 0.005 * len(covered)
        cpx[3::4] = alpha
    else:
        cpx[3::4] = [1.0] * (res * res)
    color.pixels.foreach_set(cpx)
    bpy.data.images.remove(alpha_tmp)
    for im in (color, normal, rough):
        im.pack()

    # Game material: only nodes every exporter understands
    gm = bpy.data.materials.new("%s_Game" % name)
    gm["real_snow_game"] = True
    if bpy.app.version < (5, 0, 0):
        gm.use_nodes = True
    g = gm.node_tree
    g.nodes.clear()
    out = g.nodes.new("ShaderNodeOutputMaterial"); out.location = (400, 0)
    pb = g.nodes.new("ShaderNodeBsdfPrincipled"); pb.location = (100, 0)
    tc = g.nodes.new("ShaderNodeTexImage"); tc.image = color; tc.location = (-400, 200)
    tn = g.nodes.new("ShaderNodeTexImage"); tn.image = normal; tn.location = (-400, -350)
    tr = g.nodes.new("ShaderNodeTexImage"); tr.image = rough; tr.location = (-400, -80)
    nm = g.nodes.new("ShaderNodeNormalMap"); nm.location = (-120, -350)
    nm.uv_map = uv_name
    g.links.new(tc.outputs["Color"], pb.inputs["Base Color"])
    if cutout:
        # Round = alpha cutout (glTF exports it as alphaMode MASK, cutoff 0.5),
        # which engines handle far better than blended transparency
        rnd = g.nodes.new("ShaderNodeMath"); rnd.operation = 'ROUND'; rnd.location = (-120, 120)
        g.links.new(tc.outputs["Alpha"], rnd.inputs[0])
        g.links.new(rnd.outputs[0], pb.inputs["Alpha"])
        if hasattr(gm, "surface_render_method"):
            gm.surface_render_method = 'DITHERED'
    g.links.new(tr.outputs["Color"], pb.inputs["Roughness"])
    g.links.new(tn.outputs["Color"], nm.inputs["Color"])
    g.links.new(nm.outputs["Normal"], pb.inputs["Normal"])
    g.links.new(pb.outputs["BSDF"], out.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(gm)

    # Vertex colour for engine shaders: R = thickness, G = reveal time
    mesh = obj.data
    thick = mesh.attributes.get("snow_thickness")
    reveal = mesh.attributes.get("snow_reveal")
    col = mesh.color_attributes.get("SnowData") or \
        mesh.color_attributes.new("SnowData", 'FLOAT_COLOR', 'POINT')
    vals = []
    for i in range(len(mesh.vertices)):
        t = min(1.0, max(0.0, thick.data[i].value)) if thick else 1.0
        r = reveal.data[i].value if reveal else 0.0
        vals.extend((t, r, 0.0, 1.0))
    col.data.foreach_set("color", vals)
    mesh.color_attributes.active_color = col
    obj["real_snow_game"] = True
    return True


def game_targets(context):
    out = []
    for o in context.selected_objects:
        cands = [o] if o.get("real_snow") else [c for c in o.children if c.get("real_snow")]
        for c in cands:
            if c.type == 'MESH' and c not in out and not c.get("real_snow_preview"):
                out.append(c)
    return out


class SNOW_OT_BakeGame(Operator):
    bl_idname = "snow.bake_game"
    bl_label = "Prepare Snow for Game"
    bl_description = ("Bake the snow material to textures (colour, normal, roughness), turn its "
                      "animation into morph targets and switch to a material that exports "
                      "to game engines (glTF / FBX)")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context) -> bool:
        return context.mode == 'OBJECT' and bool(game_targets(context))

    def execute(self, context):
        res = int(context.scene.snow.bake_resolution)
        done = []
        for obj in game_targets(context):
            if obj.get("real_snow_game"):
                continue
            if bake_for_game(context, obj, res, smart_uvs=GAME_SMART_UVS):
                done.append(obj.name)
        if not done:
            self.report({'WARNING'}, "Real Snow: nothing to prepare (already done?)")
            return {'CANCELLED'}
        self.report({'INFO'}, "Real Snow: prepared %d object(s) for game export" % len(done))
        return {'FINISHED'}



# ---------------------------------------------------------------------------
# Live preview
# ---------------------------------------------------------------------------

PREVIEW_DELAY = 0.35
_preview_state = {"busy": False, "msg": ""}


def clear_previews():
    for o in [o for o in bpy.data.objects if o.get("real_snow_preview")]:
        mesh = o.data
        bpy.data.objects.remove(o)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    # Show again the snow the preview was standing in for
    for o in bpy.data.objects:
        if o.get("real_snow_hidden_by_preview"):
            del o["real_snow_hidden_by_preview"]
            try:
                o.hide_set(False)
            except RuntimeError:
                pass
    # Remove the preview collection too (only if nothing else was put in it)
    coll = bpy.data.collections.get("Snow Preview")
    if coll is not None and not coll.objects and not coll.children:
        bpy.data.collections.remove(coll)


def build_previews(context):
    """Quick low-detail snow on the selection, with the real material."""
    s = context.scene.snow
    clear_previews()
    targets = snow_targets(context)
    if not targets:
        _preview_state["msg"] = "Select a mesh to preview"
        return 0
    coll = context.scene.collection.children.get("Snow Preview")
    if coll is None:
        coll = bpy.data.collections.new("Snow Preview")
        context.scene.collection.children.link(coll)
    total = 0
    cache = {}
    scene_bvh = _scene_bvh(targets, context.evaluated_depsgraph_get())
    # Add Snow would replace the snow already there: hide it meanwhile
    if s.replace_existing:
        for obj in targets:
            for child in obj.children:
                if child.get("real_snow") and child.visible_get():
                    child["real_snow_hidden_by_preview"] = True
                    child.hide_set(True)
    for obj in targets:
        snow = make_snow(context, obj, s, 'LOW', cache, scene_bvh=scene_bvh)
        if snow is None:
            continue
        snow.name = "Snow Preview"
        snow["real_snow_preview"] = True
        coll.objects.link(snow)
        snow.hide_select = True
        snow.hide_render = True
        total += len(snow.data.vertices)
    _preview_state["msg"] = "Preview: %d object(s), %d vertices (Low quality)" % (len(targets), total)
    return total


def _preview_timer():
    _preview_state["busy"] = False
    ctx = bpy.context
    scene = getattr(ctx, "scene", None)
    if scene is None or not scene.snow.live_preview or ctx.mode != 'OBJECT':
        return None
    try:
        build_previews(ctx)
    except Exception as exc:  # never break the UI from a timer
        _preview_state["msg"] = "Preview failed: %s" % exc
    for window in ctx.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
    return None


def request_preview(*_args):
    """Debounced rebuild: settings sliders fire many updates per second."""
    if bpy.app.timers.is_registered(_preview_timer):
        bpy.app.timers.unregister(_preview_timer)
    bpy.app.timers.register(_preview_timer, first_interval=PREVIEW_DELAY)


def _on_setting_change(self, context):
    if self.live_preview:
        request_preview()


def _on_preview_toggle(self, context):
    if self.live_preview:
        request_preview()
    else:
        clear_previews()


def _on_selection_change(*_args):
    scene = bpy.context.scene
    if scene is not None and scene.snow.live_preview:
        request_preview()


_msgbus_owner = object()


def _subscribe_selection():
    bpy.msgbus.clear_by_owner(_msgbus_owner)
    bpy.msgbus.subscribe_rna(key=(bpy.types.LayerObjects, "active"), owner=_msgbus_owner,
                             args=(), notify=_on_selection_change)


@persistent
def _on_save_pre(*_args):
    # Previews are never saved into the .blend
    clear_previews()


@persistent
def _on_load_post(*_args):
    clear_previews()
    _subscribe_selection()


class SNOW_OT_Preview(Operator):
    bl_idname = "snow.preview_refresh"
    bl_label = "Refresh Preview"
    bl_description = "Rebuild the snow preview for the current selection"
    bl_options = {'REGISTER'}

    def execute(self, context):
        build_previews(context)
        self.report({'INFO'}, "Real Snow: " + _preview_state["msg"])
        return {'FINISHED'}


# Properties
class SnowSettings(PropertyGroup):
    mode : EnumProperty(
        update = _on_setting_change,
        name = "Mode",
        items = (
            ('SURFACE', "Surface", "Continuous snow blanket, for ground, roofs and solid objects"),
            ('CLUMPS', "Clumps", "Separate rounded snow pillows, for pine branches, bushes and foliage"),
            ),
        default = 'SURFACE'
        )

    clump_size : FloatProperty(
        update = _on_setting_change,
        name = "Clump Size",
        description = "Radius of one snow pillow. Keep it well below the branch size: "
                      "about 1/10 to 1/15 of a branch's length",
        default = 0.15,
        min = 0.01,
        soft_max = 1.0,
        step = 1,
        precision = 3,
        subtype = 'DISTANCE'
        )

    thickness : FloatProperty(
        update = _on_setting_change,
        name = "Thickness",
        description = "How tall the pillows pile up relative to their width",
        default = 0.7,
        min = 0.0,
        max = 1.0
        )

    clump_seed : IntProperty(
        update = _on_setting_change,
        name = "Seed",
        description = "Random seed (the same seed gives the same snow)",
        default = 0,
        min = 0
        )

    use_leaf_alpha : BoolProperty(
        update = _on_setting_change,
        name = "Use Leaf Alpha",
        description = "Only put snow where the needle/leaf texture is opaque. Needed for trees "
                      "made of alpha-textured cards, otherwise snow covers the invisible parts",
        default = True
        )

    sky_occlusion : BoolProperty(
        update = _on_setting_change,
        name = "Sky Occlusion",
        description = "Little snow where something above shelters the spot: under branches, "
                      "eaves, tables",
        default = True
        )

    coverage : IntProperty(
        update = _on_setting_change,
        name = "Coverage",
        description = "Percentage of the object to be covered with snow",
        default = 100,
        min = 0,
        max = 100,
        subtype = 'PERCENTAGE'
        )

    height : FloatProperty(
        update = _on_setting_change,
        name = "Height",
        description = "Height of the snow",
        default = 0.3,
        step = 1,
        precision = 2,
        min = 0.1,
        max = 1
        )

    detail : EnumProperty(
        update = _on_setting_change,
        name = "Quality",
        description = "Mesh resolution of the snow. Lower quality gives far fewer vertices",
        items = (
            ('DRAFT', "Draft", "Very light, for blocking out and large backgrounds"),
            ('LOW', "Low", "Light"),
            ('MEDIUM', "Medium", "Balanced"),
            ('HIGH', "High", "Close-ups (same mesh resolution as Real Snow 1.4)"),
            ('ULTRA', "Ultra", "Hero shots, about 2x the vertices of High"),
            ),
        default = 'MEDIUM'
        )

    optimize_mesh : BoolProperty(
        update = _on_setting_change,
        name = "Optimize Mesh",
        description = "Remove vertices where the snow is flat and keep them where it curves. "
                      "Much lighter for the same look",
        default = True
        )

    replace_existing : BoolProperty(
        name = "Replace Existing",
        description = "Delete snow previously added to an object before adding new snow, "
                      "so you can tweak settings and press Add Snow again",
        default = True
        )

    snow_type : EnumProperty(
        update = _on_setting_change,
        name = "Snow Type",
        description = "Look of the snow material",
        items = (
            ('FRESH', "Fresh", "Bright, powdery fresh snow"),
            ('PACKED', "Packed", "Older, compacted snow: duller and greyer"),
            ('WET', "Wet", "Melting snow: glossy, slightly translucent and blue-grey"),
            ('DIRTY', "Dirty", "Roadside snow with grime and patches of dirt"),
            ),
        default = 'FRESH'
        )

    max_slope : FloatProperty(
        update = _on_setting_change,
        name = "Max Slope",
        description = "Snow slides off anything steeper than this",
        default = math.radians(60.0), min = math.radians(5.0), max = math.radians(89.0),
        subtype = 'ANGLE'
        )

    wind_direction : FloatProperty(
        update = _on_setting_change,
        name = "Wind Direction",
        description = "Direction the wind blew from: drifts stretch along it",
        default = 0.0, subtype = 'ANGLE'
        )

    drift_size : FloatProperty(
        update = _on_setting_change,
        name = "Drift Size",
        description = "Size of the drifts and wind-scoured hollows",
        default = 1.5, min = 0.05, soft_max = 20.0, step = 10, precision = 2, unit = 'LENGTH'
        )

    drift_strength : FloatProperty(
        update = _on_setting_change,
        name = "Drift Strength",
        description = "How much the depth varies between drifts and hollows (high values "
                      "scour some spots bare)",
        default = 0.4, min = 0.0, max = 1.0
        )

    wall_drifts : BoolProperty(
        update = _on_setting_change,
        name = "Wall Drifts",
        description = "Snow piles up against walls and into corners",
        default = True
        )

    wall_drift_width : FloatProperty(
        update = _on_setting_change,
        name = "Drift Width",
        description = "How far from a wall the drift reaches",
        default = 0.5, min = 0.0, soft_max = 5.0, step = 1, precision = 2, unit = 'LENGTH'
        )

    wall_drift_height : FloatProperty(
        update = _on_setting_change,
        name = "Drift Height",
        description = "How much deeper the snow is right against a wall (1 = twice as deep)",
        default = 1.0, min = 0.0, soft_max = 3.0
        )

    icicles : BoolProperty(
        name = "Icicles",
        description = "Icicles hanging from the edges of roofs and ledges that carry snow",
        default = False
        )

    icicle_length : FloatProperty(
        name = "Icicle Length",
        description = "Length of the longest icicles",
        default = 0.25, min = 0.01, soft_max = 2.0, step = 1, precision = 2, unit = 'LENGTH'
        )

    icicle_spacing : FloatProperty(
        name = "Icicle Spacing",
        description = "Average distance between icicles along an edge",
        default = 0.06, min = 0.005, soft_max = 1.0, step = 1, precision = 3, unit = 'LENGTH'
        )

    remove_hidden : BoolProperty(
        update = _on_setting_change,
        name = "Remove Hidden",
        description = "Delete the underside of the snow that is buried inside the object (about half the mesh)",
        default = True
        )

    render_displacement : BoolProperty(
        update = _on_setting_change,
        name = "Render Displacement",
        description = "Subdivide and truly displace the snow at render time (adaptive in Cycles). "
                      "Much heavier to render; when off, bump mapping is used",
        default = False
        )

    vertices : BoolProperty(
        update = _on_setting_change,
        name = "Selected Faces",
        description = "Add snow only on selected faces",
        default = False
        )

    live_preview : BoolProperty(
        name = "Live Preview",
        description = "Show a quick low-detail preview of the snow on the selected objects, "
                      "updated as you change settings (not rendered, not saved)",
        default = False,
        update = _on_preview_toggle
        )

    animation : EnumProperty(
        name = "Animation",
        description = "Animate the snow",
        items = (
            ('NONE', "None", "Static snow"),
            ('BUILD_UP', "Build Up", "Snow settles and builds up over the frame range"),
            ('MELT', "Melt", "Snow melts away over the frame range, thin snow first"),
            ),
        default = 'NONE'
        )

    anim_start : IntProperty(name = "Start Frame", description = "First frame of the animation", default = 1)

    anim_length : IntProperty(name = "Length", description = "Length of the animation in frames",
                              default = 120, min = 1)

    bake_resolution : EnumProperty(
        name = "Texture Size",
        description = "Resolution of the baked textures for game export",
        items = (('512', "512", ""), ('1024', "1024", ""), ('2048', "2048", ""), ('4096', "4096", "")),
        default = '1024'
        )

    falling_snow : BoolProperty(
        name = "Falling Snow",
        description = "Snowflakes falling onto the selection (particles from a hidden emitter above it)",
        default = False
        )

    flake_count : IntProperty(name = "Flakes", description = "Number of snowflakes over the animation",
                              default = 5000, min = 10, soft_max = 100000)

    flake_size : FloatProperty(name = "Flake Size", description = "Size of the snowflakes",
                               default = 0.006, min = 0.0005, soft_max = 0.05, step = 0.1,
                               precision = 4, unit = 'LENGTH')

    fall_speed : FloatProperty(name = "Fall Speed", description = "How fast the flakes fall, in metres per second",
                               default = 1.0, min = 0.05, soft_max = 5.0)


#############################################################################################
classes = (
    REAL_PT_snow,
    SNOW_OT_Create,
    SnowSettings,
    SNOW_OT_Preview,
    SNOW_OT_BakeGame
    )

# Register
def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.snow = PointerProperty(type=SnowSettings)
    bpy.app.handlers.save_pre.append(_on_save_pre)
    bpy.app.handlers.load_post.append(_on_load_post)
    _subscribe_selection()


# Unregister
def unregister():
    bpy.msgbus.clear_by_owner(_msgbus_owner)
    for handlers, fn in ((bpy.app.handlers.save_pre, _on_save_pre),
                         (bpy.app.handlers.load_post, _on_load_post)):
        if fn in handlers:
            handlers.remove(fn)
    if bpy.app.timers.is_registered(_preview_timer):
        bpy.app.timers.unregister(_preview_timer)
    clear_previews()
    for cls in classes:
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.snow


if __name__ == "__main__":
    register()
