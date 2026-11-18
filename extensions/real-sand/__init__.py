# SPDX-License-Identifier: GPL-2.0-or-later
#
# Real Sand: settle a layer of sand on the selected objects.
# Sister add-on of Real Snow, sharing its lightweight metaball layer.

bl_info = {
    "name": "Real Sand",
    "description": "Generate a sand layer that settles like real sand",
    "author": "Real Snow authors, adapted for sand",
    "version": (1, 7, 1),
    "blender": (4, 2, 0),
    "location": "View 3D > Sidebar > Real Sand",
    "category": "Object",
}

import math
import random

import numpy as np

import bpy
import bmesh
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty
from bpy.types import Operator, Panel, PropertyGroup
from mathutils import Vector, noise
from mathutils.kdtree import KDTree
from mathutils.bvhtree import BVHTree


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

class REAL_PT_sand(Panel):
    bl_space_type = "VIEW_3D"
    bl_context = "objectmode"
    bl_region_type = "UI"
    bl_label = "Sand"
    bl_category = "Real Sand"

    def draw(self, context):
        s = context.scene.sand
        layout = self.layout

        box = layout.box()
        row = box.row(align=True)
        row.prop(s, "live_preview", toggle=True, icon='HIDE_OFF' if s.live_preview else 'HIDE_ON')
        sub = row.row(align=True)
        sub.enabled = s.live_preview
        sub.operator("sand.preview_refresh", text="", icon='FILE_REFRESH')
        if s.live_preview and _preview_state["msg"]:
            box.label(text=_preview_state["msg"], icon='INFO')

        col = layout.column(align=True)
        col.prop(s, "coverage", slider=True)
        col.prop(s, "depth")

        layout.use_property_split = True
        layout.use_property_decorate = False

        col = layout.column(heading="Settling")
        col.prop(s, "repose_angle")
        col.prop(s, "edge_mode")
        sub = col.column()
        sub.active = s.edge_mode == 'ERODED'
        sub.prop(s, "edge_erosion")
        col.prop(s, "wall_banks")
        sub = col.column()
        sub.active = s.wall_banks
        sub.prop(s, "bank_width")

        col = layout.column(heading="Drifts")
        col.prop(s, "dune_size")
        col.prop(s, "dune_strength", slider=True)
        col.prop(s, "unevenness", slider=True)
        col.prop(s, "wind_direction")
        col.prop(s, "seed")

        col = layout.column(heading="Mesh")
        col.prop(s, "detail")
        col.prop(s, "optimize_mesh")
        col.prop(s, "only_selected")
        col.prop(s, "remove_hidden")
        col.prop(s, "replace_existing")

        col = layout.column(heading="Material")
        col.prop(s, "color_preset")
        col.prop(s, "ripples")

        box = layout.box()
        box.label(text="Animation & Effects", icon="ANIM")
        col = box.column()
        col.prop(s, "animation")
        sub = col.column(align=True)
        sub.active = s.animation != 'NONE'
        sub.prop(s, "anim_start")
        sub.prop(s, "anim_length")
        row = col.row()
        row.active = s.ripples
        row.prop(s, "moving_ripples")
        sub = col.column()
        sub.active = s.ripples and s.moving_ripples
        sub.prop(s, "ripple_speed")
        col.prop(s, "blowing_grains")
        sub = col.column(align=True)
        sub.active = s.blowing_grains
        sub.prop(s, "grain_count")
        col.prop(s, "edge_spill")
        sub = col.column(align=True)
        sub.active = s.edge_spill
        sub.prop(s, "spill_count")
        sub.prop(s, "spill_grain_size")
        sub.prop(s, "spill_piles")
        sub = col.column(align=True)
        sub.active = s.blowing_grains or s.edge_spill
        sub.prop(s, "grain_size")
        sub.prop(s, "wind_strength")

        row = layout.row(align=True)
        row.scale_y = 1.5
        row.operator("sand.create", text="Add Sand", icon="MOD_OCEAN")

        box = layout.box()
        box.label(text="Game Export", icon="EXPORT")
        col = box.column()
        col.prop(s, "bake_resolution")
        col.operator("sand.bake_game", text="Prepare for Game", icon="RENDER_STILL")


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

def sand_targets(context):
    """Selected meshes to put sand on.

    Sand made earlier stays selected after Add Sand: use the object it sits
    on instead, so tweaking settings and pressing Add Sand again re-does that
    object. Previews and effect objects are ignored.
    """
    objects = []
    for o in context.selected_objects:
        if o.type != 'MESH' or o.get("real_sand_preview") or o.get("real_sand_fx"):
            continue
        if o.get("real_sand"):
            o = o.parent
            if o is None or o.type != 'MESH':
                continue
        if o not in objects:
            objects.append(o)
    return objects


class SAND_OT_Create(Operator):
    bl_idname = "sand.create"
    bl_label = "Create Sand"
    bl_description = "Settle a layer of sand on the selected mesh objects"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context) -> bool:
        return any(o.type == 'MESH' for o in context.selected_objects)

    def execute(self, context):
        s = context.scene.sand
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        clear_previews()
        objects = sand_targets(context)
        if not objects:
            self.report({'WARNING'}, "Real Sand: select at least one mesh object")
            return {'CANCELLED'}

        if s.replace_existing:
            for obj in objects:
                for child in list(obj.children):
                    if child.get("real_sand"):
                        _remove_object(child)

        depsgraph = context.evaluated_depsgraph_get()
        # Walls the sand can bank against: every selected mesh, so sand on a
        # floor also piles up against a selected box standing on it.
        scene_bvh = _scene_bvh(objects, depsgraph)

        created, skipped, total = [], [], 0
        notes = set()
        wm = context.window_manager
        wm.progress_begin(0, len(objects))
        for i, obj in enumerate(objects):
            wm.progress_update(i)
            sand, note = build_sand(context, obj, depsgraph, s, scene_bvh)
            if note:
                notes.add(note)
            if sand is None:
                skipped.append(obj.name)
                continue
            coll = context.scene.collection.children.get("Sand")
            if coll is None:
                coll = bpy.data.collections.new("Sand")
                context.scene.collection.children.link(coll)
            coll.objects.link(sand)
            sand.data.materials.append(get_sand_material(s))
            sand.parent = obj
            sand.matrix_parent_inverse = obj.matrix_world.inverted()
            # New objects get their world matrix on the next update: the
            # animation and effects below need the real position
            context.view_layer.update()
            # The material reads these; animations keyframe them
            sand["sand_progress"] = 1.0
            sand["sand_blow"] = 0.0
            sand["sand_build"] = 0.0
            if s.animation != 'NONE':
                add_build_animation(sand, s, scene_bvh)
            if s.blowing_grains:
                add_blowing_grains(context, sand, s)
            if s.edge_spill:
                add_edge_spill(context, sand, s, scene_bvh, objects)
            created.append(sand)
            total += len(sand.data.vertices)
        wm.progress_end()

        if created:
            bpy.ops.object.select_all(action='DESELECT')
            for o in created:
                o.select_set(True)
            context.view_layer.objects.active = created[0]
        if skipped:
            self.report({'WARNING'}, "Real Sand: no sand on " + ", ".join(skipped) +
                        " (no surface flatter than the Angle of Repose" +
                        (", or no faces selected" if s.only_selected else "") +
                        ", or Coverage too low)")
        if not created:
            return {'CANCELLED'}
        msg = "Real Sand: added sand to %d object(s), %d vertices" % (len(created), total)
        if notes:
            msg += " (" + "; ".join(sorted(notes)) + ")"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

# Mesh cell size (as a fraction of element spacing) and element spacing
# (as a fraction of Depth) for each Detail level.
DETAIL_STEP = {'DRAFT': 2.4, 'LOW': 1.6, 'MEDIUM': 1.0, 'HIGH': 0.6, 'ULTRA': 0.4}
# Finest element spacing, as a fraction of the object's size (sqrt of its
# sand-holding area). Without it a thin dusting over a large floor would be
# meshed at a few millimetres: 2 cm of sand on an 8 m floor went from 31k to
# 187k vertices and 37 s. A thin, smooth layer does not need that.
SIZE_STEP = 1.0 / 150.0
# Mesh optimisation: fraction of faces kept after collapsing flat areas
OPTIMIZE_RATIO = {'DRAFT': 0.3, 'LOW': 0.4, 'MEDIUM': 0.5, 'HIGH': 0.65, 'ULTRA': 0.8}
MAX_ELEMENTS = 150000
SMOOTH_ITERATIONS = 5


def _world_bmesh(obj, depsgraph):
    mesh = bpy.data.meshes.new_from_object(obj.evaluated_get(depsgraph))
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bpy.data.meshes.remove(mesh)
    bm.transform(obj.matrix_world)
    bm.normal_update()
    return bm


def _scene_bvh(objects, depsgraph):
    verts, polys = [], []
    for obj in objects:
        bm = _world_bmesh(obj, depsgraph)
        base = len(verts)
        verts.extend(v.co.copy() for v in bm.verts)
        polys.extend(tuple(base + v.index for v in f.verts) for f in bm.faces)
        bm.free()
    return BVHTree.FromPolygons(verts, polys)


def _grid_key(p, cell):
    return (int(math.floor(p.x / cell)), int(math.floor(p.y / cell)), int(math.floor(p.z / cell)))


def sample_anchors(tris, weights, area, step, rng):
    """Evenly spread points over the triangles, about `step` apart.

    Vectorized with numpy (bundled with Blender): random area-weighted
    samples, then one kept per grid cell. This replaced a pure-Python
    Poisson-disk loop that took most of the time on large, thin layers.
    """
    gen = np.random.default_rng(rng.randrange(1 << 30))
    a = np.array([t[0] for t in tris], dtype=np.float64)
    b = np.array([t[1] for t in tris], dtype=np.float64)
    c = np.array([t[2] for t in tris], dtype=np.float64)
    nrm = np.array([t[3] for t in tris], dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    # One kept per cell: the cell size matches the density the metaball
    # field was calibrated for (and the old Poisson spacing of 0.7 * step)
    cell = step * 0.98
    count = int(min(MAX_ELEMENTS * 6, area * 4.0 / (cell * cell))) + 10
    idx = gen.choice(len(w), size=count, p=w / w.sum())
    u = gen.random(count)
    v = gen.random(count)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    pts = a[idx] + (b[idx] - a[idx]) * u[:, None] + (c[idx] - a[idx]) * v[:, None]
    keys = np.floor(pts / cell).astype(np.int64)
    # First point in each cell (the samples are already in random order)
    _, first = np.unique(keys, axis=0, return_index=True)
    first.sort()
    return [(Vector(pts[i]), Vector(nrm[idx[i]])) for i in first]


def _thin_points(points, spacing, rng):
    """Poisson-disk style thinning: keep points at least `spacing` apart."""
    grid, kept = {}, []
    order = list(range(len(points)))
    rng.shuffle(order)
    for i in order:
        p = points[i][0]
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
            kept.append(points[i])
    return kept


def _wall_bank(p, nz, bvh, width, tan_repose, rays=8):
    """Extra sand where a wall rises next to p: a ramp at the angle of repose.

    Sand that hits a wall slides down and piles up against it, so the depth
    grows the closer p is to a wall, with a slope equal to the angle of repose.
    """
    if width <= 0.0:
        return 0.0
    origin = p + Vector((0.0, 0.0, width * 0.05))
    best = 0.0
    for k in range(rays):
        a = 2.0 * math.pi * k / rays
        d = Vector((math.cos(a), math.sin(a), 0.0))
        loc, normal, _i, dist = bvh.ray_cast(origin, d, width)
        if loc is None:
            continue
        # Only steep faces facing back towards p count as walls
        if abs(normal.z) > 0.6 or normal.dot(d) > -0.2:
            continue
        best = max(best, (width - dist) * tan_repose)
    return best


def _edge_distance(p, bvh, max_dist, step, tan_repose, dirs=8):
    """Approximate horizontal distance from p to the nearest drop-off (edge).

    Probes the surface around p in several directions; returns max_dist if
    the support continues that far in every direction.
    """
    # Start the probe high enough to clear an uphill slope (up to the angle
    # of repose): starting below it, the ray found no ground and every slope
    # read as an edge on its uphill side
    lift = Vector((0.0, 0.0, step * 2.0 + max_dist * tan_repose))
    fall = lift.z + step * 2.0 + max_dist * tan_repose
    down = Vector((0.0, 0.0, -1.0))

    def supported(q, x):
        loc, _n, _i, _dist = bvh.ray_cast(q + lift, down, fall)
        # A slope gentler than the angle of repose is still support;
        # only a steeper drop counts as an edge
        return loc is not None and loc.z >= p.z - step * 0.5 - x * tan_repose

    best = max_dist
    for k in range(dirs):
        a = 2.0 * math.pi * (k + 0.5) / dirs
        direc = Vector((math.cos(a), math.sin(a), 0.0))
        if supported(p + direc * max_dist, max_dist):
            continue
        edge = 0.0
        for f in (0.125, 0.25, 0.5, 0.75):
            if supported(p + direc * (max_dist * f), max_dist * f):
                edge = max_dist * f
            else:
                break
        best = min(best, edge)
    return best


def build_sand(context, obj, depsgraph, s, scene_bvh, detail=None):
    """Return (sand object or None, note for the report)."""
    rng = random.Random(s.seed)
    depth = s.depth
    cos_repose = math.cos(s.repose_angle)
    tan_repose = math.tan(s.repose_angle)

    bm = _world_bmesh(obj, depsgraph)
    own_bvh = BVHTree.FromBMesh(bm)
    tris, weights = [], []
    for tri in bm.calc_loop_triangles():
        f = tri[0].face
        if s.only_selected and not f.select:
            continue
        nz = f.normal.z
        if nz <= cos_repose:
            continue  # too steep: sand slides off
        a, b, c = (l.vert.co.copy() for l in tri)
        area = (b - a).cross(c - a).length * 0.5
        if area > 0.0:
            tris.append((a, b, c, f.normal.copy()))
            weights.append(area)
    bm.free()
    if not tris:
        return None, ""

    # Element spacing follows the sand depth, but never finer than the
    # object's size calls for; grow it if the area is huge
    area = sum(weights)
    detail = detail or s.detail
    # (at most 3x the depth coarser, so thin layers keep their shape)
    step = max(depth * 1.2, min(math.sqrt(area) * SIZE_STEP, depth * 3.0), 0.004) * DETAIL_STEP[detail]
    note = ""
    if area / (step * step) > MAX_ELEMENTS:
        step = math.sqrt(area / MAX_ELEMENTS)
        note = "large area: mesh detail reduced"

    anchors = sample_anchors(tris, weights, area, step, rng)

    # Drifts: smooth noise stretched along the wind
    wind = Vector((math.cos(s.wind_direction), math.sin(s.wind_direction), 0.0))
    side = Vector((-wind.y, wind.x, 0.0))
    size = max(s.dune_size, 1e-3)
    seed_off = Vector((rng.uniform(-500, 500), rng.uniform(-500, 500), rng.uniform(-500, 500)))
    cover = s.coverage / 100.0
    # Coverage < 100%: patches of sand with bare ground between them
    thr = 1.0 - 2.0 * cover

    # The layer is built about one cell thicker than wanted and lowered back
    # afterwards (see lower_layer()): sand thinner than a cell (a dusting,
    # or thin sand on a slope or in a hollow) then still meshes cleanly
    extra = step * 1.15
    # Small-scale unevenness works in a frame scaled by the depth, so it shows
    # whatever the Drift Size is (scallops, little mounds, wind tails)
    fine = max(depth, 0.01)
    # Edges: ERODED = wind-stripped and thinning before the edge,
    # SLOPED = reaches the edge, sloping off at the angle of repose,
    # FULL = full depth right up to the edge
    edge_mode = s.edge_mode
    erosion = s.edge_erosion if edge_mode == 'ERODED' else 0.0
    elements = []
    up = Vector((0.0, 0.0, 1.0))
    covered_gap = max(depth, step) * 1.5
    banked = []
    for p, normal in anchors:
        nz = normal.z
        # Covered by another selected part (a panel lying on this surface)
        hit = scene_bvh.ray_cast(p + up * (step * 0.05), up, covered_gap)
        if hit[0] is not None and hit[1].z < -0.5:
            continue
        q = p + seed_off
        # Anisotropic coordinates: drifts are 2.5x longer along the wind
        dq = Vector((q.dot(wind) / (size * 2.5), q.dot(side) / size, q.z / size))
        n = noise.noise(dq) + 0.35 * noise.noise(dq * 2.7)
        fq = Vector((q.dot(wind) / (fine * 14.0), q.dot(side) / (fine * 7.0), q.z / (fine * 7.0)))
        small = 0.65 * noise.noise(fq) + 0.35 * noise.noise(fq * 2.9)
        # Ragged boundary noise, a few depths across
        rag = noise.noise(q / (fine * 2.5))
        dune = 1.0 + s.dune_strength * n
        dune *= 1.0 + s.unevenness * 1.1 * small
        # Full coverage: hollows get shallow but never bare
        dune = max(dune, 0.3 * cover)
        # Patchiness for coverage below 100%, with ragged borders
        if cover < 0.999:
            m = (n * 1.2 + 0.25 * rag - thr) / 0.5
            if m <= 0.0:
                continue
            dune *= min(1.0, m)
        # Thinner on slopes, nothing at the angle of repose
        slope = (nz - cos_repose) / max(1e-4, 1.0 - cos_repose)
        slope = min(1.0, max(0.0, slope)) ** 0.5
        d = depth * max(0.0, dune) * slope
        edge = None
        if edge_mode != 'FULL' and (d > step * 0.15 or s.wall_banks):
            reach = max(d / tan_repose, erosion, s.bank_width * tan_repose if s.wall_banks else 0.0)
            edge = _edge_distance(p, own_bvh, reach, step, tan_repose)
            # Wind strips exposed edges: thin, broken sand near them
            # (the base layer only: a bank against a wall is sheltered)
            if erosion > 0.0 and edge < erosion:
                e = edge / erosion + 0.45 * rag
                e = min(1.0, max(0.0, e))
                d *= e * e * (3.0 - 2.0 * e)
        if s.wall_banks:
            bank = _wall_bank(p, nz, scene_bvh, s.bank_width, tan_repose) * min(1.0, cover * 1.5)
            if bank > 0.0:
                banked.append(p)
                # A drift, not a wedge: its height varies along the wall and
                # it carries the same small-scale unevenness as the rest
                bq = q / max(s.bank_width * 2.5, 1e-3)
                bank *= 0.7 + 0.45 * (0.5 + 0.5 * noise.noise(bq))
                bank *= 1.0 + s.unevenness * 0.35 * small
                # Blend into the layer (soft max) instead of stacking on it
                k = max(depth, step) * 0.8
                d = 0.5 * (d + bank + math.sqrt((d - bank) ** 2 + k * k)) - 0.5 * k
        if edge is not None:
            # Sand cannot pile up over an edge: slope off at the angle of
            # repose. Applied to the total, wall banks included, so a bank
            # running to the end of a ledge slopes down instead of hanging
            # over it (which left a hollow shell)
            d = min(d, (edge + step * 0.25) * tan_repose)
        if d < step * 0.15:
            continue
        # The mesh cannot hold a layer thinner than about one cell (its top
        # dips into the surface between blobs and gets cut to pieces). Thin
        # layers are built `extra` thicker and lowered back once meshed
        d += extra
        elements.append((p, normal, d))

    if not elements:
        return None, note

    center = sum((e[0] for e in elements), Vector()) / len(elements)
    # Elements sit `sink` below the surface with a correspondingly bigger
    # reach, so the layer is always several mesh cells thick (a thin layer
    # would be polygonized with holes). The buried half is removed later.
    sink = step * 0.8
    name = "RealSandTmp%06d" % random.randint(0, 999999)
    ball = bpy.data.metaballs.new(name)
    # Sand is smooth, so the mesh can be coarser than the element spacing
    cell = step * 0.8
    ball.resolution = cell
    ball.render_resolution = ball.resolution
    ball.threshold = 1.0
    # Calibrated: with this radius and threshold the layer's top sits about
    # `d` above the anchors and fades to nothing at d = 0.
    for co, normal, d in elements:
        el = ball.elements.new()
        el.co = co - normal * sink - center
        el.radius = step * 0.8 + d + sink
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
        return None, note

    max_d = max(e[2] for e in elements)
    finish_mesh(mesh, center, scene_bvh, s.remove_hidden, cell,
                reach=max_d + step * 0.8 + sink + cell * 4.0, full_edges=edge_mode == 'FULL')
    if len(mesh.vertices) == 0:
        bpy.data.meshes.remove(mesh)
        return None, note
    if extra > step * 0.45:
        # Not quite all the way: a thin base stays everywhere the sand is,
        # so shallow hollows read as thin sand rather than bare patches
        lower_layer(mesh, center, scene_bvh, extra - step * 0.45, cell)
        if len(mesh.vertices) == 0:
            bpy.data.meshes.remove(mesh)
            return None, note
    if s.wall_banks and s.bank_width > 0.0:
        if banked:
            shape_banks(mesh, center, scene_bvh, own_bvh, s, cell, seed_off, edge_mode,
                        banked, step)
    if s.optimize_mesh:
        optimize_mesh(mesh, OPTIMIZE_RATIO[detail])
    # Full edges: thin sand at the rim should read as solid, not fade out
    store_thickness(mesh, center, own_bvh, depth * (0.35 if edge_mode == 'FULL' else 1.0))
    mesh.name = "Sand"
    sand = bpy.data.objects.new("Sand", mesh)
    sand["real_sand"] = True
    sand.location = center
    return sand, note


# ---------------------------------------------------------------------------
# Animation & effects
# ---------------------------------------------------------------------------

FRONT_SOFTNESS = 0.15   # width of the moving front, as a fraction of the sweep


def _reveal_times(sand, s):
    """Per-vertex time (0..1) at which the front reaches each vertex.

    The front sweeps downwind: Build Up deposits sand from the upwind side
    onward, Blow Away strips it from the upwind side onward, thin sand first.
    """
    mesh = sand.data
    mw = sand.matrix_world
    wind = Vector((math.cos(s.wind_direction), math.sin(s.wind_direction), 0.0))
    pts = [mw @ v.co for v in mesh.vertices]
    along = [p.dot(wind) for p in pts]
    lo, hi = min(along), max(along)
    span = max(hi - lo, 1e-6)
    attr = mesh.attributes.get("sand_thickness")
    thick = [min(1.0, max(0.0, d.value)) for d in attr.data] if attr else [1.0] * len(pts)
    scale = max(s.dune_size * 0.3, s.depth * 6.0)
    times = []
    for p, a, t in zip(pts, along, thick):
        x = (a - lo) / span
        jitter = noise.noise(p / scale) * 0.12
        if s.animation == 'BLOW_AWAY':
            tau = 0.75 * x + 0.2 * t + jitter        # thin sand goes first
        else:
            tau = 0.75 * x + 0.2 * (1.0 - t) + jitter  # thin edges fill in last
        times.append(min(1.0, max(0.0, tau / 0.95)))
    return times


def add_build_animation(sand, s, scene_bvh):
    """Build Up / Blow Away: a soft front sweeping downwind over the frame range.

    Geometry: the smooth Geometry Nodes animation (every vertex rises or sinks
    on its own schedule). Material: grains appear / vanish as the front
    passes (object property "sand_progress" against "sand_reveal").
    """
    start = s.anim_start
    length = max(1, s.anim_length)
    mesh = sand.data
    mw = sand.matrix_world
    inv = mw.inverted()
    down = Vector((0.0, 0.0, -1.0))
    sink = max(0.002, s.depth * 0.1)
    blow = s.animation == 'BLOW_AWAY'

    times = _reveal_times(sand, s)
    flat = []
    for v in mesh.vertices:
        p = mw @ v.co
        loc, _n, _i, _d = scene_bvh.ray_cast(p + Vector((0, 0, 1e-4)), down, 10.0)
        if loc is None:
            loc, _n, _i, _d = scene_bvh.find_nearest(p)
        # Just below the surface: hidden by the object itself
        flat.append(inv @ (loc + down * sink) if loc is not None else v.co.copy())
    add_anim_modifier(sand, times, flat, softness=0.3, invert=blow, start=start, length=length)

    sand["sand_blow"] = 1.0 if blow else 0.0
    sand["sand_build"] = 0.0 if blow else 1.0
    sand["sand_progress"] = 0.0
    sand.keyframe_insert('["sand_progress"]', frame=start)
    sand["sand_progress"] = 1.0
    sand.keyframe_insert('["sand_progress"]', frame=start + length)
    _linear_keys(sand)


def _grain_object(s):
    """Tiny low-poly grain used as the particle instance."""
    name = "Sand Grain"
    obj = bpy.data.objects.get(name)
    if obj is None or obj.type != 'MESH':
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=1, radius=1.0)
        mesh = bpy.data.meshes.new(name)
        bm.to_mesh(mesh)
        bm.free()
        obj = bpy.data.objects.new(name, mesh)
        obj["real_sand_fx"] = True
        coll = bpy.data.collections.get("Sand Assets")
        if coll is None:
            coll = bpy.data.collections.new("Sand Assets")
            bpy.context.scene.collection.children.link(coll)
        coll.objects.link(obj)
        # Keep the source grain out of sight; its instances still render
        obj.location = (0.0, 0.0, -1000.0)
        obj.hide_set(True)
    mat_name = "Sand Grain %s" % s.color_preset.title()
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(mat_name)
        if bpy.app.version < (5, 0, 0):
            mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        dark, light = SAND_COLORS[s.color_preset]
        bsdf.inputs["Base Color"].default_value = tuple((dark[i] + light[i]) * 0.5 for i in range(3)) + (1.0,)
        bsdf.inputs["Roughness"].default_value = 0.9
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return obj


def add_blowing_grains(context, sand, s):
    """Particles: a few grains hop off exposed spots and are carried downwind.

    Only the sand's own particles move: the wind is their launch velocity
    plus Brownian jitter, not a scene force field, so nothing else in the
    scene is pushed around. Grains die after a short hop.
    """
    grain = _grain_object(s)
    wind = Vector((math.cos(s.wind_direction), math.sin(s.wind_direction), 0.0))
    mw = sand.matrix_world
    # Emit from exposed spots: thick sand on slopes facing the wind, and crests
    vg = sand.vertex_groups.new(name="sand_emit")
    attr = sand.data.attributes.get("sand_thickness")
    normals = [mw.to_3x3() @ v.normal for v in sand.data.vertices]
    for i, n in enumerate(normals):
        t = min(1.0, max(0.0, attr.data[i].value)) if attr else 1.0
        n = n.normalized()
        facing = max(0.0, -n.dot(wind))       # slope facing into the wind
        top = max(0.0, n.z) ** 4               # flat tops / crests
        vg.add([i], t * min(1.0, 0.25 * top + 1.5 * facing), 'REPLACE')
    mod = sand.modifiers.new("Blowing Sand", 'PARTICLE_SYSTEM')
    psys = mod.particle_system
    psys.vertex_group_density = vg.name
    psys.seed = s.seed
    st = psys.settings
    st.name = "Blowing Sand"
    scene = context.scene
    if s.animation == 'BLOW_AWAY':
        st.frame_start = s.anim_start
        st.frame_end = s.anim_start + max(1, s.anim_length)
    else:
        st.frame_start = scene.frame_start
        st.frame_end = scene.frame_end
    st.count = s.grain_count
    fps = scene.render.fps / max(scene.render.fps_base, 1e-6)
    st.lifetime = max(4, int(fps * 0.8))       # short hops
    st.lifetime_random = 0.6
    st.emit_from = 'FACE'
    st.use_emit_random = True
    st.physics_type = 'NEWTON'
    st.mass = 0.001
    # Launch: downwind (in the emitter's local axes) plus a small lift
    local_wind = mw.inverted().to_3x3() @ (wind * s.wind_strength)
    st.object_align_factor = local_wind
    st.normal_factor = 0.25 * min(1.0, s.wind_strength)
    st.factor_random = 0.2 * s.wind_strength
    st.brownian_factor = 0.2 * s.wind_strength
    st.drag_factor = 0.2
    st.effector_weights.gravity = 0.4
    # Ignore any force fields in the scene; only gravity
    st.effector_weights.all = 0.0
    st.effector_weights.gravity = 0.4
    st.render_type = 'OBJECT'
    st.instance_object = grain
    st.particle_size = s.grain_size
    st.size_random = 0.7
    st.display_method = 'DOT'
    st.display_percentage = 25


def _edge_direction(p, bvh, reach, step, tan_repose, dirs=12):
    """(distance, horizontal direction) to the nearest drop-off edge within reach."""
    lift = Vector((0.0, 0.0, step * 2.0 + reach * tan_repose))
    down = Vector((0.0, 0.0, -1.0))
    fall = lift.z + step * 2.0 + reach * tan_repose

    def supported(q, x):
        loc, _n, _i, _d = bvh.ray_cast(q + lift, down, fall)
        return loc is not None and loc.z >= p.z - step * 0.5 - x * tan_repose

    best = (reach, None)
    for k in range(dirs):
        a = 2.0 * math.pi * k / dirs
        d = Vector((math.cos(a), math.sin(a), 0.0))
        if supported(p + d * reach, reach):
            continue
        edge = 0.0
        for f in (0.1, 0.2, 0.35, 0.5, 0.7, 0.85):
            if supported(p + d * (reach * f), reach * f):
                edge = reach * f
            else:
                break
        if edge < best[0]:
            best = (edge, d)
    return best


def add_edge_spill(context, sand, s, scene_bvh, targets):
    """Particles: sand trickling over the edges in a few streams and falling
    down the sides.

    Grains are emitted from a small hidden emitter placed just *outside* the
    edges (emitted on top of the object they would fall into it), at the
    spots where a noise field picks a stream.
    """
    for child in list(sand.children):
        if child.get("real_sand_fx"):
            _remove_object(child)
    mw = sand.matrix_world
    tan_repose = math.tan(s.repose_angle)
    reach = max(s.depth * 3.0, 0.06)
    if s.edge_mode == 'ERODED':
        # Eroded sand stops short of the edge: look further for it
        reach = max(reach, s.edge_erosion + 0.06)
    step = max(s.depth * 0.5, 0.01)
    attr = sand.data.attributes.get("sand_thickness")
    patch = max(0.012, s.depth * 0.3)
    # Every spot along the edges where sand could spill over
    candidates = {}
    for i, v in enumerate(sand.data.vertices):
        t = attr.data[i].value if attr else 1.0
        if t < 0.02:
            continue
        p = mw @ v.co
        key = (round(p.x / patch), round(p.y / patch), round(p.z / patch))
        if key in candidates:
            continue
        edge, d = _edge_direction(p, scene_bvh, reach, step, tan_repose)
        if d is None:
            continue
        lip = p + d * (edge + patch * 0.6)
        candidates[key] = (lip, d, (1.0 - edge / reach))
    # Streams: a few concentrated pour points spread along all the edges
    # (few dense streams read far better than many thin ones), each a small
    # cluster of spots
    rng = random.Random(s.seed + 11)
    spacing = max(0.3, s.depth * 8.0)
    order = list(candidates.values())
    rng.shuffle(order)
    centres = []
    for lip, d, w in order:
        if all((lip - c).length >= spacing * rng.uniform(0.8, 1.3) for c in centres):
            centres.append(lip)
    spread = patch * 2.5
    spots = {}
    for key, (lip, d, w) in candidates.items():
        near = min(((lip - c).length for c in centres), default=spread)
        if near < spread:
            spots[key] = (lip, d, w * (1.0 - near / spread))
    if not spots:
        return False
    # Emitter: one small upright patch per spot, facing outwards
    bm = bmesh.new()
    for lip, d, w in spots.values():
        side = Vector((-d.y, d.x, 0.0)) * (patch * 0.5)
        upv = Vector((0.0, 0.0, patch * 0.5))
        vs = [bm.verts.new(lip + a + b) for a, b in ((-side, -upv), (side, -upv), (side, upv), (-side, upv))]
        f = bm.faces.new(vs)
        f.normal_update()
        if f.normal.dot(d) < 0:
            f.normal_flip()
    mesh = bpy.data.meshes.new("Sand Spill Emitter")
    bm.to_mesh(mesh)
    bm.free()
    em = bpy.data.objects.new("Sand Spill Emitter", mesh)
    em["real_sand_fx"] = True
    for c in sand.users_collection:
        c.objects.link(em)
    em.parent = sand
    em.matrix_parent_inverse = sand.matrix_world.inverted()
    em.display_type = 'WIRE'
    em.hide_select = True
    em.show_instancer_for_render = False
    em.show_instancer_for_viewport = False

    zs = [(o.matrix_world @ Vector(c)).z for o in targets for c in o.bound_box]
    top = max(lip.z for lip, _d, _w in spots.values())
    fall = max(0.2, top - min(zs) + 0.05)
    scene = context.scene
    fps = scene.render.fps / max(scene.render.fps_base, 1e-6)
    mod = em.modifiers.new("Edge Spill", 'PARTICLE_SYSTEM')
    psys = mod.particle_system
    psys.seed = s.seed + 7
    st = psys.settings
    st.name = "Edge Spill"
    if s.animation == 'BLOW_AWAY':
        st.frame_start = s.anim_start
        st.frame_end = s.anim_start + max(1, s.anim_length)
    else:
        st.frame_start = scene.frame_start
        st.frame_end = scene.frame_end
    st.count = s.spill_count
    st.lifetime = max(4, int(math.sqrt(2.0 * fall / 9.81) * fps * 1.15) + 2)
    st.lifetime_random = 0.1
    st.emit_from = 'FACE'
    st.use_emit_random = True
    st.physics_type = 'NEWTON'
    st.mass = 0.001
    # A gentle push out over the edge, then gravity
    st.normal_factor = 0.12
    st.factor_random = 0.03
    st.brownian_factor = 0.01
    st.effector_weights.all = 0.0
    st.effector_weights.gravity = 1.0
    st.render_type = 'OBJECT'
    # Falling grains blur into streaks: a stretched grain aligned with its
    # velocity reads as a continuous stream without needing huge counts
    st.instance_object = _streak_object(s)
    st.particle_size = s.spill_grain_size
    st.size_random = 0.5
    st.use_rotations = True
    st.rotation_mode = 'VEL'
    st.use_dynamic_rotation = True
    # Show the real grains in the viewport too (dots are hard to see)
    st.display_method = 'RENDER'
    # Viewport shows a fifth of the grains (renders use all of them)
    st.display_percentage = 20
    if s.spill_piles:
        _add_spill_piles(context, sand, s, spots, patch, st.frame_start, st.frame_end)
    return True


def _streak_object(s):
    """A grain stretched along X (particles align X with their velocity)."""
    name = "Sand Streak"
    obj = bpy.data.objects.get(name)
    if obj is None or obj.type != 'MESH':
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=1, radius=1.0)
        bmesh.ops.scale(bm, vec=(7.0, 0.8, 0.8), verts=bm.verts)
        mesh = bpy.data.meshes.new(name)
        bm.to_mesh(mesh)
        bm.free()
        for poly in mesh.polygons:
            poly.use_smooth = True
        obj = bpy.data.objects.new(name, mesh)
        obj["real_sand_fx"] = True
        coll = bpy.data.collections.get("Sand Assets")
        if coll is None:
            coll = bpy.data.collections.new("Sand Assets")
            bpy.context.scene.collection.children.link(coll)
        coll.objects.link(obj)
        obj.location = (0.0, 0.0, -1000.0)
        obj.hide_set(True)
    grain = _grain_object(s)
    obj.data.materials.clear()
    obj.data.materials.append(grain.data.materials[0])
    return obj


def _add_spill_piles(context, sand, s, spots, patch, f0, f1):
    """A small heap grows where each stream lands (if something is below)."""
    tan_repose = math.tan(s.repose_angle)
    down = Vector((0.0, 0.0, -1.0))
    # Group neighbouring spots into streams
    streams = {}
    cell = patch * 5.0
    for lip, d, w in spots.values():
        key = (round(lip.x / cell), round(lip.y / cell), round(lip.z / cell))
        streams.setdefault(key, []).append((lip, d, w))
    depsgraph = context.evaluated_depsgraph_get()
    scene = context.scene
    mat = get_sand_material(s)
    made = 0
    for members in streams.values():
        lip = sum((m[0] for m in members), Vector()) / len(members)
        out = sum((m[1] for m in members), Vector()).normalized()
        strength = sum(m[2] for m in members)
        origin = lip + out * patch
        # Land on the first real surface below (not sand, previews or effects)
        hit = None
        for _ in range(6):
            ok, loc, nrm, _i, ob, _m = scene.ray_cast(depsgraph, origin, down)
            if not ok:
                break
            if ob.get("real_sand") or ob.get("real_sand_fx") or ob.get("real_sand_preview"):
                origin = loc + down * 1e-3
                continue
            hit = (loc, nrm)
            break
        if hit is None or hit[1].z < 0.5:
            continue
        radius = min(0.35, max(0.07, patch * 3.0 + 0.03 * strength))
        height = radius * tan_repose * 0.8
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=False, segments=28, radius1=radius,
                              radius2=radius * 0.08, depth=height)
        bmesh.ops.translate(bm, verts=bm.verts, vec=(0.0, 0.0, height * 0.5))
        bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=3, use_grid_fill=True)
        rng = random.Random(made + s.seed)
        for v in bm.verts:
            if v.co.z > 1e-4:
                n = noise.noise(v.co / max(radius * 0.6, 1e-3) + Vector((made * 7.3, 0, 0)))
                v.co.z *= 1.0 + 0.25 * n
                v.co.x *= 1.0 + 0.1 * rng.uniform(-1, 1)
        for _ in range(2):
            bmesh.ops.smooth_vert(bm, verts=bm.verts[:], factor=0.5,
                                  use_axis_x=True, use_axis_y=True, use_axis_z=True)
        mesh = bpy.data.meshes.new("Sand Spill Pile")
        bm.to_mesh(mesh)
        bm.free()
        for poly in mesh.polygons:
            poly.use_smooth = True
        # Solid sand, fading out into grains at the foot
        th = mesh.attributes.new("sand_thickness", 'FLOAT', 'POINT')
        th.data.foreach_set("value", [min(1.0, v.co.z / max(height * 0.25, 1e-6))
                                      for v in mesh.vertices])
        mesh.materials.append(mat)
        pile = bpy.data.objects.new("Sand Spill Pile", mesh)
        pile["real_sand_fx"] = True
        pile["sand_progress"] = 1.0
        for c in sand.users_collection:
            c.objects.link(pile)
        pile.location = hit[0]
        pile.parent = sand
        pile.matrix_parent_inverse = sand.matrix_world.inverted()
        # Grows while the sand pours: the volume grows steadily, so the size
        # grows fast at first and then slows (cube root)
        for t in (0.0, 0.05, 0.2, 0.45, 0.7, 1.0):
            k = max(0.05, t ** (1.0 / 3.0))
            pile.scale = (k, k, k)
            pile.keyframe_insert("scale", frame=f0 + t * (f1 - f0))
        made += 1
    return made


def _remove_object(obj):
    """Remove an object made by the add-on, with its own effect children."""
    for child in list(obj.children):
        if child.get("real_sand_fx"):
            _remove_object(child)
    data = obj.data
    bpy.data.objects.remove(obj)
    if data is not None and data.users == 0 and isinstance(data, bpy.types.Mesh):
        bpy.data.meshes.remove(data)


def lower_layer(mesh, center, scene_bvh, extra, cell):
    """Lower a layer that was built `extra` too thick back to its real depth.

    Every vertex moves down towards the surface below it by `extra`; where
    the layer was thinner than that it ends up on the surface, and faces
    lying entirely on the surface are removed (that is where thin sand runs
    out). Drifts and banks keep their shape: everything moves by the same
    amount.
    """
    up = Vector((0.0, 0.0, 1.0))
    down = Vector((0.0, 0.0, -1.0))
    bm = bmesh.new()
    bm.from_mesh(mesh)
    floor = cell * 0.02
    grounded = set()
    reach = extra * 4.0 + cell * 8.0
    for v in bm.verts:
        p = v.co + center
        loc, _n, _i, dist = scene_bvh.ray_cast(p + up * 1e-4, down, reach)
        if loc is None:
            continue
        h = p.z - loc.z
        if h - extra <= floor:
            v.co.z = loc.z + floor * 0.5 - center.z
            grounded.add(v)
        else:
            v.co.z -= extra
    gone = [f for f in bm.faces if all(v in grounded for v in f.verts)]
    if gone:
        bmesh.ops.delete(bm, geom=gone, context='FACES_ONLY')
        loose = [v for v in bm.verts if not v.link_faces]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context='VERTS')
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def optimize_mesh(mesh, ratio):
    """Collapse-decimate the mesh: flat areas lose most of their vertices,
    curved areas (drifts, banks, edges) keep theirs."""
    if ratio >= 0.999 or len(mesh.polygons) < 200:
        return
    tmp = bpy.data.objects.new("RealSandOptimize", mesh)
    bpy.context.scene.collection.objects.link(tmp)
    try:
        mod = tmp.modifiers.new("Optimize", 'DECIMATE')
        mod.decimate_type = 'COLLAPSE'
        mod.ratio = ratio
        mod.use_collapse_triangulate = False
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()
        reduced = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg))
    finally:
        bpy.data.objects.remove(tmp)
    # Swap the geometry into the original mesh datablock
    bm = bmesh.new()
    bm.from_mesh(reduced)
    bm.to_mesh(mesh)
    bm.free()
    bpy.data.meshes.remove(reduced)
    for poly in mesh.polygons:
        poly.use_smooth = True
    mesh.update()


def shape_banks(mesh, center, scene_bvh, own_bvh, s, cell, seed_off, edge_mode,
                banked=None, step=0.0):
    """Lift the sand surface onto a proper drift against walls.

    The metaball layer rounds off anything smaller than its blobs, so a bank
    comes out as a flat-topped plateau that never climbs the wall. Here every
    upward-facing vertex near a wall is raised to the bank profile: rising at
    the angle of repose to a crisp contact line on the wall, its height
    varying along the wall like a real drift.
    """
    width = s.bank_width
    tan_repose = math.tan(s.repose_angle)
    up = Vector((0.0, 0.0, 1.0))
    down = Vector((0.0, 0.0, -1.0))
    cover = min(1.0, s.coverage / 100.0 * 1.5)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.normal_update()
    moved = []
    near_wall = None
    if banked:
        # Only vertices near a spot that got a bank can be against a wall:
        # skips the wall search (12 rays) for the rest of the floor
        near_wall = KDTree(len(banked))
        for i, q in enumerate(banked):
            near_wall.insert(q, i)
        near_wall.balance()
        near_reach = step * 2.0 + cell * 2.0
    for v in bm.verts:
        if v.normal.z < 0.2 or v.is_boundary:
            continue
        p = v.co + center
        if near_wall is not None and near_wall.find(p)[2] > near_reach:
            continue
        floor, _n, _i, _d = own_bvh.ray_cast(p + up * 1e-3, down, width * 3.0 + 1.0)
        if floor is None:
            continue
        # Distance to the nearest wall, measured just above the floor
        origin = Vector((p.x, p.y, floor.z + width * 0.05))
        dist = width
        for k in range(12):
            a = 2.0 * math.pi * k / 12
            d = Vector((math.cos(a), math.sin(a), 0.0))
            loc, normal, _i, hd = scene_bvh.ray_cast(origin, d, width)
            if loc is None or abs(normal.z) > 0.6 or normal.dot(d) > -0.2:
                continue
            dist = min(dist, hd)
        if dist >= width:
            continue
        q = p + seed_off
        bq = q / max(width * 2.5, 1e-3)
        var = 0.7 + 0.45 * (0.5 + 0.5 * noise.noise(bq))
        target = floor.z + (width - dist) * tan_repose * var * cover
        if p.z < target:
            v.co.z = target - center.z
            moved.append(v)
    if moved:
        # Settle the lifted surface a little (keeps the drift, loses facets)
        ring = set(moved)
        for v in moved:
            for e in v.link_edges:
                ring.add(e.other_vert(v))
        ring = [v for v in ring if not v.is_boundary]
        for _ in range(2):
            bmesh.ops.smooth_vert(bm, verts=ring, factor=0.4,
                                  use_axis_x=False, use_axis_y=False, use_axis_z=True)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def store_thickness(mesh, center, bvh, depth):
    """Per-vertex sand thickness (1.0 = the Depth setting), for the material.

    The material uses it to fade thin sand out grain by grain, so the sand
    ends in a scatter of grains instead of a hard rim.
    """
    attr = mesh.attributes.get("sand_thickness")
    if attr is None:
        attr = mesh.attributes.new("sand_thickness", 'FLOAT', 'POINT')
    values = []
    for v in mesh.vertices:
        p = v.co + center
        loc, normal, _i, dist = bvh.find_nearest(p)
        if loc is None:
            values.append(1.0)
            continue
        h = (p - loc).dot(normal)
        values.append(max(0.0, h) / depth)
    attr.data.foreach_set("value", values)


def finish_mesh(mesh, center, scene_bvh, remove_hidden, cell, reach, full_edges=False):
    """Smooth the sand and cut away everything that should not be there.

    The metaball layer extends below the surface and past edges. Instead of
    trusting face normals (which fails for open meshes, thin panels and
    models built from several parts), every vertex is tested with rays
    against all selected objects:
      * a surface right above it  -> buried under/inside geometry: removed
      * no surface below it       -> hanging past an edge: removed
    """
    bm = bmesh.new()
    bm.from_mesh(mesh)
    up = Vector((0.0, 0.0, 1.0))
    down = Vector((0.0, 0.0, -1.0))
    eps = cell * 0.02
    buried = {}
    hanging = set()
    for v in bm.verts:
        p = v.co + center
        if remove_hidden:
            loc, normal, _i, _d = scene_bvh.ray_cast(p + up * eps, up, reach)
            # Buried = inside: the first surface above faces up, so we are
            # looking at it from behind (inside an object / under a floor).
            # Under an overhang (that surface faces down) is not buried.
            if loc is not None and normal.z > 0.0:
                buried[v] = True
                continue
        loc, _n, _i, _d = scene_bvh.ray_cast(p, down, reach)
        if loc is None:
            hanging.add(v)
    # Faces touching a buried corner go, as do wholly hanging faces
    dead = [f for f in bm.faces
            if any(v in buried for v in f.verts) or all(v in hanging for v in f.verts)]
    if dead:
        bmesh.ops.delete(bm, geom=dead, context='FACES_ONLY')
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context='VERTS')
    # The cut leaves an open border hovering just above the surface: lower it
    # straight down onto the surface, so the sand meets it without a gap
    for v in bm.verts:
        if v.is_boundary and v not in hanging:
            p = v.co + center
            loc, _n, _i, _d = scene_bvh.ray_cast(p + up * eps, down, cell * 3.0)
            if loc is not None:
                v.co = loc - center
    # Spread that drop into a gentle slope (border stays pinned on the
    # surface), so the sand thins out towards its edge
    inner = [v for v in bm.verts if not v.is_boundary and v not in hanging]
    for _ in range(4):
        bmesh.ops.smooth_vert(bm, verts=inner, factor=0.5,
                              use_axis_x=False, use_axis_y=False, use_axis_z=True)
    # Faces crossing an edge: put their outer corners on the edge itself
    # (not down the side), so the sand ends at the edge
    for v in hanging:
        if not v.is_valid:
            continue
        p = v.co + center
        near, normal, _i, _d = scene_bvh.find_nearest(p)
        if near is None:
            continue
        if abs(normal.z) < 0.7:
            # Nearest is a side face: find the top edge above it
            inward = Vector((-normal.x, -normal.y, 0.0))
            if inward.length > 1e-6:
                inward.normalize()
            probe = near + inward * (cell * 0.05) + up * reach
            top, _n, _i, _d = scene_bvh.ray_cast(probe, down, reach * 2.0)
            if top is not None:
                near = top - inward * (cell * 0.05)
                if full_edges:
                    # Full edges: move sideways onto the edge and keep the
                    # height, so the sand ends in a straight face at the edge
                    near = Vector((near.x, near.y, max(p.z, top.z)))
        v.co = near - center
    # Sand is smooth, not blobby: relax the surface. Done once the buried
    # underside is gone and the overhang is on the edge, with the border and
    # the edge held: smoothing the closed shell pulled top and bottom together
    # (a 1 cm layer sank into the surface and was deleted as buried), and
    # smoothing towards unsnapped overhang dragged sand down the sides
    inner = [v for v in bm.verts
             if v.is_valid and not v.is_boundary and v not in hanging]
    for _ in range(SMOOTH_ITERATIONS):
        bmesh.ops.smooth_vert(bm, verts=inner, factor=0.5,
                              use_axis_x=True, use_axis_y=True, use_axis_z=True)
    # Snapped faces that ended up standing upright on the edge show as dark
    # slivers: drop them (with full edges that upright face is the point)
    bm.normal_update()
    upright = [] if full_edges else [
        f for f in bm.faces if abs(f.normal.z) < 0.35 and any(v in hanging for v in f.verts)]
    if upright:
        bmesh.ops.delete(bm, geom=upright, context='FACES_ONLY')
        loose = [v for v in bm.verts if not v.link_faces]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context='VERTS')
    # Faces lying flat in the surface itself (every corner lowered or snapped
    # onto it) have no thickness and z-fight with the object: drop them
    on_surface = set()
    for v in bm.verts:
        loc, _n, _i, dist = scene_bvh.ray_cast(v.co + center + up * eps * 2.0, down, eps * 6.0)
        if loc is not None:
            on_surface.add(v)
    flat = [f for f in bm.faces if all(v in on_surface for v in f.verts)]
    if flat:
        bmesh.ops.delete(bm, geom=flat, context='FACES_ONLY')
        loose = [v for v in bm.verts if not v.link_faces]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context='VERTS')
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    for poly in mesh.polygons:
        poly.use_smooth = True
    mesh.update()


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------

MATERIAL_VERSION = 6

SAND_COLORS = {
    'DESERT': ((0.62, 0.42, 0.22), (0.78, 0.58, 0.34)),
    'BEACH': ((0.70, 0.60, 0.44), (0.86, 0.77, 0.60)),
    'WHITE': ((0.80, 0.78, 0.72), (0.93, 0.91, 0.86)),
    'DARK': ((0.10, 0.09, 0.08), (0.22, 0.20, 0.18)),
    'RED': ((0.45, 0.17, 0.07), (0.62, 0.29, 0.13)),
}


def get_sand_material(s):
    name = "Sand %s%s%s" % (s.color_preset.title(), " Ripples" if s.ripples else "",
                            " Moving" if (s.ripples and s.moving_ripples) else "")
    mat = bpy.data.materials.get(name)
    if mat is not None and mat.get("real_sand") == MATERIAL_VERSION:
        return mat
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat["real_sand"] = MATERIAL_VERSION
    if bpy.app.version < (5, 0, 0):
        mat.use_nodes = True
    # Thin sand fades out grain by grain through alpha
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = 'DITHERED'
    elif hasattr(mat, "blend_method"):
        mat.blend_method = 'HASHED'
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    nodes.clear()

    def node(kind, x, y, **props):
        n = nodes.new(kind)
        n.location = (x, y)
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def math_node(op, x, y, a=None, b=None, clamp=False):
        n = node("ShaderNodeMath", x, y, operation=op, use_clamp=clamp)
        if a is not None:
            n.inputs[0].default_value = a
        if b is not None:
            n.inputs[1].default_value = b
        return n

    out = node("ShaderNodeOutputMaterial", 900, 0)
    bsdf = node("ShaderNodeBsdfPrincipled", 550, 0)
    coord = node("ShaderNodeTexCoord", -1500, 0)

    # --- Thickness: thin sand -> grains drop out
    thick = node("ShaderNodeAttribute", -1300, 700, attribute_name="sand_thickness")
    patch = node("ShaderNodeTexNoise", -1300, 500)
    patch.inputs["Scale"].default_value = 25.0
    patch.inputs["Detail"].default_value = 4.0
    # fill = thickness / 0.45 + (patch noise - 0.5) * 0.6
    fill_a = math_node('DIVIDE', -1050, 700, b=0.45)
    links.new(thick.outputs["Fac"], fill_a.inputs[0])
    fill_b = math_node('SUBTRACT', -1050, 500, b=0.5)
    links.new(patch.outputs["Fac"], fill_b.inputs[0])
    fill_c = math_node('MULTIPLY_ADD', -850, 600, b=0.6)
    links.new(fill_b.outputs[0], fill_c.inputs[0])
    fill_c.inputs[1].default_value = 0.6
    links.new(fill_a.outputs[0], fill_c.inputs[2])
    # Animation: a front sweeps over the sand. x = how far the front has
    # passed this point (0..1); Build Up shows x, Blow Away shows 1 - x.
    #   x = clamp((progress * (1 + w) - reveal) / w)
    # Static sand has progress = 1 and reveal = 0, so it is fully shown.
    progress = node("ShaderNodeAttribute", -1700, 1100, attribute_type='OBJECT',
                    attribute_name="sand_progress")
    blow = node("ShaderNodeAttribute", -1700, 900, attribute_type='OBJECT',
                attribute_name="sand_blow")
    reveal = node("ShaderNodeAttribute", -1700, 1300, attribute_name="sand_reveal")
    w = FRONT_SOFTNESS
    fx_a = math_node('MULTIPLY_ADD', -1500, 1100)
    links.new(progress.outputs["Fac"], fx_a.inputs[0])
    fx_a.inputs[1].default_value = 1.0 + w
    links.new(reveal.outputs["Fac"], fx_a.inputs[2])
    fx_a.inputs[2].default_value = 0.0
    # (progress*(1+w)) - reveal: MULTIPLY_ADD adds, so negate reveal first
    neg = math_node('MULTIPLY', -1500, 1300, b=-1.0)
    links.new(reveal.outputs["Fac"], neg.inputs[0])
    links.new(neg.outputs[0], fx_a.inputs[2])
    fx = math_node('DIVIDE', -1300, 1100, b=w, clamp=True)
    links.new(fx_a.outputs[0], fx.inputs[0])
    # amount = 1 - blow * x - build * (1 - x)
    # Sand without these properties (static, or made by older versions) has
    # blow = build = 0 and stays fully visible.
    build = node("ShaderNodeAttribute", -1700, 700, attribute_type='OBJECT',
                 attribute_name="sand_build")
    blow_term = math_node('MULTIPLY', -1100, 1000)
    links.new(blow.outputs["Fac"], blow_term.inputs[0])
    links.new(fx.outputs[0], blow_term.inputs[1])
    one_minus_x = math_node('SUBTRACT', -1100, 850, a=1.0)
    links.new(fx.outputs[0], one_minus_x.inputs[1])
    build_term = math_node('MULTIPLY', -950, 850)
    links.new(build.outputs["Fac"], build_term.inputs[0])
    links.new(one_minus_x.outputs[0], build_term.inputs[1])
    amount_a = math_node('SUBTRACT', -950, 1050, a=1.0)
    links.new(blow_term.outputs[0], amount_a.inputs[1])
    amount = math_node('SUBTRACT', -800, 1050, clamp=True)
    links.new(amount_a.outputs[0], amount.inputs[0])
    links.new(build_term.outputs[0], amount.inputs[1])
    amount_curve = math_node('POWER', -650, 1050, b=0.5)
    links.new(amount.outputs[0], amount_curve.inputs[0])
    fill_amt = math_node('MULTIPLY', -650, 800)
    links.new(fill_c.outputs[0], fill_amt.inputs[0])
    links.new(amount_curve.outputs[0], fill_amt.inputs[1])
    # amount 0 must hide every grain, even where the sand is thick
    fill_gate = math_node('SUBTRACT', -500, 800)
    links.new(fill_amt.outputs[0], fill_gate.inputs[0])
    gate = math_node('LESS_THAN', -650, 950, b=0.02)
    links.new(amount.outputs[0], gate.inputs[0])
    gate_big = math_node('MULTIPLY', -500, 950, b=10.0)
    links.new(gate.outputs[0], gate_big.inputs[0])
    links.new(gate_big.outputs[0], fill_gate.inputs[1])
    # One random value per grain (~2 mm cells)
    grains = node("ShaderNodeTexVoronoi", -1050, 250)
    grains.inputs["Scale"].default_value = 450.0
    links.new(coord.outputs["Object"], grains.inputs["Vector"])
    links.new(coord.outputs["Object"], patch.inputs["Vector"])
    grain_val = node("ShaderNodeSeparateColor", -850, 250)
    links.new(grains.outputs["Color"], grain_val.inputs["Color"])
    alpha = math_node('GREATER_THAN', -350, 600)
    links.new(fill_gate.outputs[0], alpha.inputs[0])
    links.new(grain_val.outputs[0], alpha.inputs[1])
    links.new(alpha.outputs[0], bsdf.inputs["Alpha"])

    # --- Colour: broad tone variation, per-grain variation, thin = darker
    tone = node("ShaderNodeTexNoise", -1050, 0)
    tone.inputs["Scale"].default_value = 2.5
    tone.inputs["Detail"].default_value = 8.0
    tone.inputs["Roughness"].default_value = 0.6
    links.new(coord.outputs["Object"], tone.inputs["Vector"])
    ramp = node("ShaderNodeValToRGB", -850, 0)
    dark, light = SAND_COLORS[s.color_preset]
    ramp.color_ramp.elements[0].color = (*dark, 1.0)
    ramp.color_ramp.elements[0].position = 0.25
    ramp.color_ramp.elements[1].color = (*light, 1.0)
    ramp.color_ramp.elements[1].position = 0.75
    mid = ramp.color_ramp.elements.new(0.5)
    mid.color = tuple(0.5 * (dark[i] + light[i]) for i in range(3)) + (1.0,)
    links.new(tone.outputs["Fac"], ramp.inputs["Fac"])
    # Per-grain brightness jitter (mixed mineral grains)
    jitter = node("ShaderNodeMapRange", -650, 250)
    jitter.inputs["To Min"].default_value = 0.72
    jitter.inputs["To Max"].default_value = 1.1
    links.new(grain_val.outputs[1], jitter.inputs["Value"])
    col_grain = node("ShaderNodeMix", -450, 50, data_type='RGBA', blend_type='MULTIPLY')
    col_grain.inputs["Factor"].default_value = 1.0
    links.new(ramp.outputs["Color"], col_grain.inputs["A"])
    links.new(jitter.outputs["Result"], col_grain.inputs["B"])
    # A few dark grains
    dark_grain = math_node('LESS_THAN', -650, 100, b=0.06)
    links.new(grain_val.outputs[2], dark_grain.inputs[0])
    col_dark = node("ShaderNodeMix", -250, 50, data_type='RGBA', blend_type='MULTIPLY')
    links.new(dark_grain.outputs[0], col_dark.inputs["Factor"])
    links.new(col_grain.outputs["Result"], col_dark.inputs["A"])
    col_dark.inputs["B"].default_value = (0.25, 0.22, 0.2, 1.0)
    # Thin sand looks a little darker and greyer (less depth to scatter light)
    thin = node("ShaderNodeMapRange", -650, 450)
    thin.inputs["From Min"].default_value = 0.0
    thin.inputs["From Max"].default_value = 0.8
    thin.inputs["To Min"].default_value = 0.75
    thin.inputs["To Max"].default_value = 1.0
    links.new(thick.outputs["Fac"], thin.inputs["Value"])
    col_thin = node("ShaderNodeMix", -50, 50, data_type='RGBA', blend_type='MULTIPLY')
    col_thin.inputs["Factor"].default_value = 1.0
    links.new(col_dark.outputs["Result"], col_thin.inputs["A"])
    links.new(thin.outputs["Result"], col_thin.inputs["B"])
    links.new(col_thin.outputs["Result"], bsdf.inputs["Base Color"])

    # --- Sparkle: a few quartz grains catch the light
    sparkle = math_node('GREATER_THAN', -450, -200, b=0.985)
    links.new(grain_val.outputs[2], sparkle.inputs[0])
    rough = node("ShaderNodeMapRange", -250, -200)
    rough.inputs["To Min"].default_value = 0.95
    rough.inputs["To Max"].default_value = 0.25
    links.new(sparkle.outputs[0], rough.inputs["Value"])
    links.new(rough.outputs["Result"], bsdf.inputs["Roughness"])
    bsdf.inputs["Specular IOR Level"].default_value = 0.35

    # --- Bump: grains, plus wind ripples
    grain_bump = node("ShaderNodeBump", 250, -350)
    grain_bump.inputs["Strength"].default_value = 0.35
    grain_bump.inputs["Distance"].default_value = 0.001
    links.new(grains.outputs["Distance"], grain_bump.inputs["Height"])
    if s.ripples:
        mapping = node("ShaderNodeMapping", -1050, -450)
        mapping.inputs["Rotation"].default_value[2] = s.wind_direction
        links.new(coord.outputs["Object"], mapping.inputs["Vector"])
        if s.moving_ripples:
            # Ripples creep downwind (simple expression: no Python needed)
            fc = mapping.inputs["Location"].driver_add("default_value", 0)
            fc.driver.type = 'SCRIPTED'
            fc.driver.expression = "frame * %.6f" % (s.ripple_speed / 24.0)
        wave = node("ShaderNodeTexWave", -850, -450, wave_type='BANDS', wave_profile='SAW')
        wave.inputs["Scale"].default_value = 12.0  # ~8 cm between crests
        # Long, gently wavy parallel crests (too much distortion looks woven)
        wave.inputs["Distortion"].default_value = 1.2
        wave.inputs["Detail"].default_value = 0.0
        wave.inputs["Detail Scale"].default_value = 1.0
        links.new(mapping.outputs["Vector"], wave.inputs["Vector"])
        # Ripples fade out where the sand is thin
        ripple_amt = math_node('MULTIPLY', -650, -450, clamp=True)
        links.new(wave.outputs["Fac"], ripple_amt.inputs[0])
        links.new(fill_a.outputs[0], ripple_amt.inputs[1])
        ripple_bump = node("ShaderNodeBump", 50, -450)
        ripple_bump.inputs["Strength"].default_value = 0.6
        ripple_bump.inputs["Distance"].default_value = 0.01
        links.new(ripple_amt.outputs[0], ripple_bump.inputs["Height"])
        links.new(ripple_bump.outputs["Normal"], grain_bump.inputs["Normal"])
    links.new(grain_bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


GAME_SMART_UVS = False   # sand is a mostly flat layer: top-down UVs are clean

# ---------------------------------------------------------------------------
# Smooth animation (Geometry Nodes) and game export
# ---------------------------------------------------------------------------
#
# The animation is a Geometry Nodes modifier: every vertex has its own reveal
# time (attribute "sand_reveal") and an offset that presses it into the surface
# (attribute "sand_flat_offset"). One keyframed "Progress" input moves a soft
# front across the mesh, so each vertex moves smoothly on its own schedule.
# Game engines cannot read Geometry Nodes, so Prepare for Game samples the
# animation into morph targets (shape keys) that blend from one sample to
# the next.

ANIM_MOD = "Sand Animation"
ANIM_GROUP = "Real Sand Animation"
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
    reveal.inputs["Name"].default_value = "sand_reveal"
    offset = N.new("GeometryNodeInputNamedAttribute"); offset.location = (-300, -400)
    offset.data_type = 'FLOAT_VECTOR'
    offset.inputs["Name"].default_value = "sand_flat_offset"
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
    reveal = mesh.attributes.get("sand_reveal") or mesh.attributes.new("sand_reveal", 'FLOAT', 'POINT')
    reveal.data.foreach_set("value", times)
    off = mesh.attributes.get("sand_flat_offset") or \
        mesh.attributes.new("sand_flat_offset", 'FLOAT_VECTOR', 'POINT')
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
    obj["sand_anim_range"] = (float(start), float(start + length))
    obj["sand_anim_invert"] = 1.0 if invert else 0.0
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
    start, end = obj.get("sand_anim_range", (1.0, 121.0))
    invert = obj.get("sand_anim_invert", 0.0) > 0.5
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
        key = obj.shape_key_add(name="%s %02d" % ("Sand", j), from_mix=False)
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


FULL_STATE_PROPS = {"sand_progress": 1.0, "sand_blow": 0.0, "sand_melt": 0.0, "sand_build": 0.0}


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
    gm["real_sand_game"] = True
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
    thick = mesh.attributes.get("sand_thickness")
    reveal = mesh.attributes.get("sand_reveal")
    col = mesh.color_attributes.get("SandData") or \
        mesh.color_attributes.new("SandData", 'FLOAT_COLOR', 'POINT')
    vals = []
    for i in range(len(mesh.vertices)):
        t = min(1.0, max(0.0, thick.data[i].value)) if thick else 1.0
        r = reveal.data[i].value if reveal else 0.0
        vals.extend((t, r, 0.0, 1.0))
    col.data.foreach_set("color", vals)
    mesh.color_attributes.active_color = col
    obj["real_sand_game"] = True
    return True


def game_targets(context):
    out = []
    for o in context.selected_objects:
        cands = [o] if o.get("real_sand") else [c for c in o.children if c.get("real_sand")]
        for c in cands:
            if c.type == 'MESH' and c not in out and not c.get("real_sand_preview"):
                out.append(c)
    return out


class SAND_OT_BakeGame(Operator):
    bl_idname = "sand.bake_game"
    bl_label = "Prepare Sand for Game"
    bl_description = ("Bake the sand material to textures (colour, normal, roughness), turn its "
                      "animation into morph targets and switch to a material that exports "
                      "to game engines (glTF / FBX)")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context) -> bool:
        return context.mode == 'OBJECT' and bool(game_targets(context))

    def execute(self, context):
        res = int(context.scene.sand.bake_resolution)
        done = []
        for obj in game_targets(context):
            if obj.get("real_sand_game"):
                continue
            if bake_for_game(context, obj, res, smart_uvs=GAME_SMART_UVS):
                done.append(obj.name)
        if not done:
            self.report({'WARNING'}, "Real Sand: nothing to prepare (already done?)")
            return {'CANCELLED'}
        self.report({'INFO'}, "Real Sand: prepared %d object(s) for game export" % len(done))
        return {'FINISHED'}



# ---------------------------------------------------------------------------
# Live preview
# ---------------------------------------------------------------------------

PREVIEW_DELAY = 0.35
_preview_state = {"msg": ""}


def clear_previews():
    for o in [o for o in bpy.data.objects if o.get("real_sand_preview")]:
        mesh = o.data
        bpy.data.objects.remove(o)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    # Show again the sand the preview was standing in for
    for o in bpy.data.objects:
        if o.get("real_sand_hidden_by_preview"):
            del o["real_sand_hidden_by_preview"]
            try:
                o.hide_set(False)
            except RuntimeError:
                pass
    # Remove the preview collection too (only if nothing else was put in it)
    coll = bpy.data.collections.get("Sand Preview")
    if coll is not None and not coll.objects and not coll.children:
        bpy.data.collections.remove(coll)


def build_previews(context):
    """Quick low-detail sand on the selection, with the real material."""
    s = context.scene.sand
    clear_previews()
    objects = sand_targets(context)
    if not objects:
        _preview_state["msg"] = "Select a mesh to preview"
        return 0
    coll = context.scene.collection.children.get("Sand Preview")
    if coll is None:
        coll = bpy.data.collections.new("Sand Preview")
        context.scene.collection.children.link(coll)
    depsgraph = context.evaluated_depsgraph_get()
    scene_bvh = _scene_bvh(objects, depsgraph)
    total = 0
    for obj in objects:
        # Add Sand would replace the sand already there: hide it meanwhile
        if s.replace_existing:
            for child in obj.children:
                if child.get("real_sand") and child.visible_get():
                    child["real_sand_hidden_by_preview"] = True
                    child.hide_set(True)
        sand, _note = build_sand(context, obj, depsgraph, s, scene_bvh, detail='LOW')
        if sand is None:
            continue
        sand.name = "Sand Preview"
        sand["real_sand_preview"] = True
        sand.pop("real_sand", None)
        coll.objects.link(sand)
        sand.data.materials.append(get_sand_material(s))
        sand.parent = obj
        sand.matrix_parent_inverse = obj.matrix_world.inverted()
        sand.hide_select = True
        sand.hide_render = True
        total += len(sand.data.vertices)
    _preview_state["msg"] = "Preview: %d object(s), %d vertices (Low quality)" % (len(objects), total)
    return total


def _preview_timer():
    ctx = bpy.context
    scene = getattr(ctx, "scene", None)
    if scene is None or not scene.sand.live_preview or ctx.mode != 'OBJECT':
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
    if scene is not None and scene.sand.live_preview:
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


class SAND_OT_Preview(Operator):
    bl_idname = "sand.preview_refresh"
    bl_label = "Refresh Preview"
    bl_description = "Rebuild the sand preview for the current selection"
    bl_options = {'REGISTER'}

    def execute(self, context):
        build_previews(context)
        self.report({'INFO'}, "Real Sand: " + _preview_state["msg"])
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class SandSettings(PropertyGroup):
    coverage: IntProperty(
        update=_on_setting_change,
        name="Coverage",
        description="How much of the surface is covered. Below 100% the sand breaks into patches",
        default=100, min=0, max=100, subtype='PERCENTAGE')

    depth: FloatProperty(
        update=_on_setting_change,
        name="Depth",
        description="Average depth of the sand layer on flat ground",
        default=0.05, min=0.002, soft_max=1.0, step=1, precision=3, unit='LENGTH')

    repose_angle: FloatProperty(
        update=_on_setting_change,
        name="Angle of Repose",
        description="Steepest slope sand can rest on. Steeper faces stay bare and banks "
                    "against walls rise at this angle (dry sand is about 34°)",
        default=math.radians(34.0), min=math.radians(5.0), max=math.radians(60.0),
        subtype='ANGLE')

    wall_banks: BoolProperty(
        update=_on_setting_change,
        name="Bank Against Walls",
        description="Pile sand up against walls and into corners",
        default=True)

    bank_width: FloatProperty(
        update=_on_setting_change,
        name="Bank Width",
        description="How far from a wall the sand bank reaches",
        default=0.3, min=0.0, soft_max=5.0, step=1, precision=2, unit='LENGTH')

    dune_size: FloatProperty(
        update=_on_setting_change,
        name="Drift Size",
        description="Size of the drifts and hollows in the sand",
        default=1.0, min=0.01, soft_max=20.0, step=10, precision=2, unit='LENGTH')

    unevenness: FloatProperty(
        update=_on_setting_change,
        name="Unevenness",
        description="Small mounds, scallops and wind tails on the surface, independent of Drift Size",
        default=0.6, min=0.0, max=1.0)

    edge_mode: EnumProperty(
        update=_on_setting_change,
        name="Edges",
        description="How the sand meets the edges of the object",
        items=(
            ('ERODED', "Eroded", "Wind strips exposed edges: the sand thins out and breaks up "
                                 "before the edge (Edge Erosion sets how far)"),
            ('SLOPED', "Sloped", "The sand reaches the edge, sloping off at the angle of repose"),
            ('FULL', "Full", "Full depth right up to the edge, like a filled tray or damp sand"),
        ),
        default='ERODED')

    edge_erosion: FloatProperty(
        update=_on_setting_change,
        name="Edge Erosion",
        description="Wind strips sand off exposed edges: within this distance of an edge the "
                    "sand thins out and breaks up",
        default=0.25, min=0.0, soft_max=2.0, step=1, precision=2, unit='LENGTH')

    animation: EnumProperty(
        name="Animation",
        description="Animate the sand settling or eroding",
        items=(
            ('NONE', "None", "Static sand"),
            ('BUILD_UP', "Build Up", "Sand accumulates from nothing over the frame range"),
            ('BLOW_AWAY', "Blow Away", "Sand is eroded away by the wind over the frame range"),
        ),
        default='NONE')

    anim_start: IntProperty(
        name="Start Frame", description="First frame of the animation", default=1)

    anim_length: IntProperty(
        name="Length", description="Length of the animation in frames", default=120, min=1)

    moving_ripples: BoolProperty(
        name="Moving Ripples",
        description="Wind ripples slowly creep downwind (animated material)",
        default=False)

    ripple_speed: FloatProperty(
        name="Ripple Speed",
        description="How fast the ripples move, in texture units per second",
        default=0.05, min=0.0, soft_max=1.0)

    blowing_grains: BoolProperty(
        name="Blowing Grains",
        description="Particles: a few grains hop off exposed spots and are carried "
                    "downwind (does not affect anything else in the scene)",
        default=False)

    grain_count: IntProperty(
        name="Grains",
        description="Number of blowing grains over the animation",
        default=3000, min=10, soft_max=100000)

    edge_spill: BoolProperty(
        name="Edge Spill",
        description="Particles: grains trickling off the edges and falling down the sides",
        default=False)

    spill_count: IntProperty(
        name="Spill Grains",
        description="Number of grains spilling off the edges over the animation",
        default=60000, min=10, soft_max=300000)

    spill_grain_size: FloatProperty(
        name="Spill Grain Size",
        description="Size of the spilling grains. Bigger than real grains reads better "
                    "from a distance",
        default=0.008, min=0.0005, soft_max=0.05, step=0.1, precision=4, unit='LENGTH')

    spill_piles: BoolProperty(
        name="Spill Piles",
        description="A small heap of sand grows where each stream lands",
        default=True)

    grain_size: FloatProperty(
        name="Grain Size",
        description="Size of the blowing grains. Real grains are under 2 mm; bigger reads "
                    "better from a distance",
        default=0.004, min=0.0002, soft_max=0.05, step=0.1, precision=4, unit='LENGTH')

    wind_strength: FloatProperty(
        name="Wind Strength",
        description="Wind speed for the grains, in metres per second",
        default=2.0, min=0.0, soft_max=20.0)

    bake_resolution: EnumProperty(
        name="Texture Size",
        description="Resolution of the baked textures for game export",
        items=(('512', "512", ""), ('1024', "1024", ""), ('2048', "2048", ""), ('4096', "4096", "")),
        default='1024')

    replace_existing: BoolProperty(
        name="Replace Existing",
        description="Delete sand previously added to an object before adding new sand, "
                    "so you can tweak settings and press Add Sand again",
        default=True)

    dune_strength: FloatProperty(
        update=_on_setting_change,
        name="Drift Strength",
        description="How much the depth varies between drifts and hollows",
        default=0.6, min=0.0, max=1.0)

    wind_direction: FloatProperty(
        update=_on_setting_change,
        name="Wind Direction",
        description="Direction the wind blew from: drifts stretch along it and ripples run across it",
        default=0.0, subtype='ANGLE')

    seed: IntProperty(
        update=_on_setting_change,name="Seed", description="Random seed", default=0, min=0)

    detail: EnumProperty(
        update=_on_setting_change,
        name="Quality",
        description="Mesh resolution of the sand",
        items=(
            ('DRAFT', "Draft", "Very light, for blocking out and large backgrounds"),
            ('LOW', "Low", "Light"),
            ('MEDIUM', "Medium", "Balanced"),
            ('HIGH', "High", "Close-ups, about 3x the vertices of Medium"),
            ('ULTRA', "Ultra", "Hero shots, about 6x the vertices of Medium"),
        ),
        default='MEDIUM')

    optimize_mesh: BoolProperty(
        update=_on_setting_change,
        name="Optimize Mesh",
        description="Remove vertices where the sand is flat and keep them where it curves "
                    "(drifts, banks, edges). Much lighter for the same look",
        default=True)

    only_selected: BoolProperty(
        update=_on_setting_change,
        name="Selected Faces",
        description="Add sand only on the faces selected in Edit Mode",
        default=False)

    remove_hidden: BoolProperty(
        update=_on_setting_change,
        name="Remove Hidden",
        description="Delete the underside of the sand that is buried inside the object",
        default=True)

    color_preset: EnumProperty(
        update=_on_setting_change,
        name="Color",
        items=(
            ('DESERT', "Desert", "Warm orange-brown desert sand"),
            ('BEACH', "Beach", "Light beige beach sand"),
            ('WHITE', "White", "Pale white sand"),
            ('DARK', "Volcanic", "Dark volcanic sand"),
            ('RED', "Red", "Red iron-rich sand"),
        ),
        default='DESERT')

    ripples: BoolProperty(
        update=_on_setting_change,
        name="Wind Ripples",
        description="Add small wind ripples to the material",
        default=True)

    live_preview: BoolProperty(
        name="Live Preview",
        description="Show a quick low-detail preview of the sand on the selected objects, "
                    "updated as you change settings (not rendered, not saved)",
        default=False,
        update=_on_preview_toggle)


classes = (REAL_PT_sand, SAND_OT_Create, SAND_OT_BakeGame, SAND_OT_Preview, SandSettings)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.sand = PointerProperty(type=SandSettings)
    bpy.app.handlers.save_pre.append(_on_save_pre)
    bpy.app.handlers.load_post.append(_on_load_post)
    _subscribe_selection()


def unregister():
    bpy.msgbus.clear_by_owner(_msgbus_owner)
    for handlers, fn in ((bpy.app.handlers.save_pre, _on_save_pre),
                         (bpy.app.handlers.load_post, _on_load_post)):
        if fn in handlers:
            handlers.remove(fn)
    if bpy.app.timers.is_registered(_preview_timer):
        bpy.app.timers.unregister(_preview_timer)
    clear_previews()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.sand


if __name__ == "__main__":
    register()
