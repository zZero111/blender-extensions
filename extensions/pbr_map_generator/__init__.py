# SPDX-License-Identifier: GPL-3.0-or-later
"""
PBR Map Generator for Blender.

* Turn any image (file on disk, image already in Blender, or a URL) into a set
  of PBR maps: Base Color, Height/Bump, Normal, Roughness and Ambient Occlusion.
* Generate procedural, seamlessly tiling textures (bricks, tiles, hex tiles,
  wood planks, cobblestones, concrete, rock, metal plates, fabric) with the
  same set of maps.
* Builds a ready-to-use Principled BSDF material and can assign it to the
  active object.

Find it in the 3D Viewport sidebar (N key) under the "PBR Maps" tab.
"""

bl_info = {
    "name": "PBR Map Generator",
    "author": "PBR Map Generator",
    "version": (1, 6, 0),
    "blender": (3, 6, 0),
    "location": "3D Viewport > Sidebar (N) > PBR Maps",
    "description": "Generate normal, bump, roughness and AO maps from images, plus procedural textures",
    "category": "Material",
}

import math
import os
import struct
import tempfile
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import urllib.parse
import urllib.request

import bmesh
import bpy
import gpu
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)

SQRT3 = 3 ** 0.5


# ---------------------------------------------------------------------------
# Image / array helpers
# ---------------------------------------------------------------------------

def image_to_array(img):
    """Return image pixels as float32 array (H, W, 4). Row 0 is the bottom."""
    w, h = img.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    return buf.reshape(h, w, 4)


def replace_datablock(old, new, name):
    """Swap `new` in everywhere `old` is used, delete `old`, and give `new` its name."""
    if old is not None and old != new:
        old.user_remap(new)
        collection = bpy.data.images if isinstance(old, bpy.types.Image) else bpy.data.materials
        collection.remove(old)
    new.name = name
    return new


def array_to_image(name, arr, non_color, out_dir="", keep=True, replace=False):
    """Create a Blender image from a (H, W) or (H, W, 3/4) float array.
    keep=False makes a throwaway image (used for previews).
    replace=True overwrites an existing image of the same name (all users update)."""
    arr = np.clip(arr.astype(np.float32), 0.0, 1.0)
    h, w = arr.shape[:2]
    if arr.ndim == 2:
        rgba = np.dstack([arr, arr, arr, np.ones_like(arr)])
    elif arr.shape[2] == 3:
        rgba = np.dstack([arr, np.ones((h, w), np.float32)])
    else:
        rgba = arr

    img = bpy.data.images.new(name, w, h, alpha=True, float_buffer=False)
    img.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    img.pixels.foreach_set(np.ascontiguousarray(rgba, dtype=np.float32).ravel())
    img.update()

    if not keep:
        return img
    if replace:
        old = bpy.data.images.get(name)
        replace_datablock(old if old != img else None, img, name)
    if out_dir:
        folder = bpy.path.abspath(out_dir)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, bpy.path.clean_name(img.name) + ".png")
        img.filepath_raw = path
        img.file_format = "PNG"
        img.save()
        img.source = "FILE"
        img.reload()
    else:
        img.pack()
    return img


_MAP_SPEC = (
    ("albedo", "_BaseColor", False),
    ("height", "_Height", True),
    ("normal", "_Normal", True),
    ("roughness", "_Roughness", True),
    ("ao", "_AO", True),
    ("metallic", "_Metallic", True),
)


def _to_u8(arr):
    """Float [0,1] -> 8-bit exactly the way Blender converts pixels for 8-bit
    images (so the saved texture is identical to setting img.pixels)."""
    v = np.clip(np.asarray(arr), 0.0, 1.0).astype(np.float32)
    u = np.floor(v * np.float32(255.0) + np.float32(0.5))
    u[v > np.float32(1.0 - 0.5 / 255.0)] = 255
    return u.astype(np.uint8)


def png_bytes(arr, level=2):
    """Encode a (H, W) grey or (H, W, 3) colour array as a PNG (pure numpy + zlib).
    Row 0 of the array is the bottom of the image, as in Blender."""
    u8 = _to_u8(arr if arr.ndim == 2 or arr.shape[2] == 3 else arr[..., :3])
    h, w = u8.shape[:2]
    ch = 1 if u8.ndim == 2 else 3
    rows = np.ascontiguousarray(u8[::-1]).reshape(h, w * ch)  # PNG stores the top row first
    raw = np.empty((h, w * ch + 1), np.uint8)
    raw[:, 0] = 2  # "Up" filter: store each row as the difference from the row above
    raw[0, 1:] = rows[0]
    np.subtract(rows[1:], rows[:-1], out=raw[1:, 1:])
    comp = zlib.compress(raw.tobytes(), level)  # zlib releases the GIL: runs in parallel

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0 if ch == 1 else 2, 0, 0, 0))
            + chunk(b"IDAT", comp) + chunk(b"IEND", b""))


def encode_maps(arrays):
    """Encode every map to PNG bytes in parallel. Safe to run off the main thread."""
    keys = [k for k, _s, _n in _MAP_SPEC if k in arrays]
    if _POOL["pool"] is None:
        _POOL["pool"] = ThreadPoolExecutor(max_workers=_workers(), thread_name_prefix="pbrgen")
    jobs = {k: _POOL["pool"].submit(png_bytes, arrays[k]) for k in keys}
    return {k: j.result() for k, j in jobs.items()}


def images_from_encoded(base, encoded, out_dir="", replace=False):
    """Main thread only: turn encoded PNGs into Blender images (packed, or saved
    to out_dir), optionally replacing existing images of the same name."""
    result = {}
    for key, suffix, non_color in _MAP_SPEC:
        if key not in encoded:
            continue
        name = base + suffix
        data = encoded[key]
        img = bpy.data.images.new(name, 8, 8)  # reserves a unique name
        if replace:
            old = bpy.data.images.get(name)
            replace_datablock(old if old != img else None, img, name)
        if out_dir:
            folder = bpy.path.abspath(out_dir)
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, bpy.path.clean_name(img.name) + ".png")
            with open(path, "wb") as f:
                f.write(data)
            img.source = "FILE"
            img.filepath = path
            img.reload()
        else:
            img.filepath_raw = "//" + bpy.path.clean_name(img.name) + ".png"
            img.pack(data=data, data_len=len(data))
            img.source = "FILE"
        img.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
        result[key] = img
    return result


def make_images(base, arrays, out_dir="", keep=True, replace=False):
    if not keep:  # throwaway preview images: small, keep the simple path
        return {key: array_to_image(base + suffix, arrays[key], non_color, out_dir, keep, replace)
                for key, suffix, non_color in _MAP_SPEC if key in arrays}
    return images_from_encoded(base, encode_maps(arrays), out_dir, replace)


def luminance(rgb):
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


_POOL = {"pool": None}


def _workers():
    return max(1, min(8, (os.cpu_count() or 2)))


def par_rows(n, fn):
    """Run fn(slice) over horizontal strips 0..n in parallel threads.
    numpy releases the GIL during array maths, so strips really run at the same
    time. Each pixel is computed exactly as before, just on a different core."""
    k = _workers()
    if n < 512 or k == 1:
        fn(slice(0, n))
        return
    if _POOL["pool"] is None:
        _POOL["pool"] = ThreadPoolExecutor(max_workers=k, thread_name_prefix="pbrgen")
    edges = np.linspace(0, n, k + 1).astype(int)
    jobs = [_POOL["pool"].submit(fn, slice(edges[i], edges[i + 1])) for i in range(k) if edges[i + 1] > edges[i]]
    for j in jobs:
        j.result()


def box_blur(a, r):
    """Fast separable box blur with wrap-around edges (keeps maps tileable)."""
    r = int(r)
    if r < 1:
        return a
    for axis in (0, 1):
        n = a.shape[axis]
        rr = min(r, max(1, n // 2 - 1))
        pad = [(0, 0)] * a.ndim
        pad[axis] = (rr + 1, rr)
        p = np.pad(a, pad, mode="wrap")
        if a.ndim != 2:
            c = np.cumsum(p, axis=axis, dtype=np.float64)
            a = ((c[2 * rr + 1:2 * rr + 1 + n] - c[0:n]) / (2 * rr + 1)).astype(np.float32) if axis == 0 else \
                ((c[:, 2 * rr + 1:2 * rr + 1 + n] - c[:, 0:n]) / (2 * rr + 1)).astype(np.float32)
            continue
        out = np.empty(a.shape, np.float32)

        # Columns (axis 0 pass) or rows (axis 1 pass) are independent, so split them across cores.
        def work(sl, p=p, n=n, rr=rr, axis=axis, out=out):
            if axis == 0:
                c = np.cumsum(p[:, sl], axis=0, dtype=np.float64)
                out[:, sl] = ((c[2 * rr + 1:2 * rr + 1 + n] - c[0:n]) / (2 * rr + 1)).astype(np.float32)
            else:
                c = np.cumsum(p[sl], axis=1, dtype=np.float64)
                out[sl] = ((c[:, 2 * rr + 1:2 * rr + 1 + n] - c[:, 0:n]) / (2 * rr + 1)).astype(np.float32)

        par_rows(a.shape[1 - axis], work)
        a = out
    return a


def gauss_blur(a, r):
    """Approximate gaussian blur: three box blur passes."""
    if r < 1:
        return a
    r3 = max(1, int(round(r / 1.7)))
    for _ in range(3):
        a = box_blur(a, r3)
    return a


def normalize01(a):
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-6:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def apply_contrast(a, contrast):
    return np.clip((a - 0.5) * contrast + 0.5, 0.0, 1.0)


def height_to_normal(height, strength, directx=False):
    """Sobel-filtered tangent-space normal map from a height map."""
    h = height
    H, W = h.shape
    # Pad once with wrap-around and take shifted *views* (no copies). Each view
    # equals np.roll(np.roll(h, dy, 0), dx, 1), so the result is unchanged.
    hp = np.pad(h, 1, mode="wrap")
    out = np.empty((H, W, 3), dtype=np.result_type(h, np.float32))

    def work(sl):
        r0, r1 = sl.start, sl.stop

        def r(_a, dy, dx):
            return hp[1 - dy + r0:1 - dy + r1, 1 - dx:1 - dx + W]

        # Note: row index increases upwards in Blender images.
        gx = (r(h, 1, -1) + 2 * r(h, 0, -1) + r(h, -1, -1)
              - r(h, 1, 1) - 2 * r(h, 0, 1) - r(h, -1, 1)) / 8.0
        gy = (r(h, -1, 1) + 2 * r(h, -1, 0) + r(h, -1, -1)
              - r(h, 1, 1) - 2 * r(h, 1, 0) - r(h, 1, -1)) / 8.0
        nx = -gx * strength
        ny = -gy * strength
        nz = np.ones_like(gx)
        length = np.sqrt(nx * nx + ny * ny + nz * nz)
        nx, ny, nz = nx / length, ny / length, nz / length
        if directx:
            ny = -ny
        out[sl, :, 0] = nx * 0.5 + 0.5
        out[sl, :, 1] = ny * 0.5 + 0.5
        out[sl, :, 2] = nz * 0.5 + 0.5

    par_rows(H, work)
    return out


def height_to_ao(height, radius, strength):
    """Cheap cavity-style ambient occlusion from a height map."""
    surround = gauss_blur(height, radius)
    cavity = np.clip(surround - height, 0.0, None)
    return np.clip(1.0 - cavity * strength, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Procedural building blocks (all tileable)
# ---------------------------------------------------------------------------

def value_noise(h, w, cells_y, cells_x, rng):
    """Smooth, tileable value noise in [0, 1]."""
    cells_y, cells_x = max(1, int(cells_y)), max(1, int(cells_x))
    grid = rng.random((cells_y, cells_x)).astype(np.float32)
    y = (np.arange(h, dtype=np.float32) + 0.5) * cells_y / h
    x = (np.arange(w, dtype=np.float32) + 0.5) * cells_x / w
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    ty = y - y0
    tx = x - x0
    ty = ty * ty * (3 - 2 * ty)
    tx = tx * tx * (3 - 2 * tx)
    y0 %= cells_y
    x0 %= cells_x
    y1 = (y0 + 1) % cells_y
    x1 = (x0 + 1) % cells_x
    # Interpolate along x once per grid row (small), then pick rows and
    # interpolate along y. Same arithmetic, in the same order, as doing it per
    # pixel, but avoids four full-size gathers.
    g0 = grid[:, x0]
    rows = g0 + (grid[:, x1] - g0) * tx[None, :]
    # Keep the same precision the per-pixel version used (it ends up float64).
    out = np.empty((h, w), np.result_type(rows, ty))

    def work(sl):
        top = rows[y0[sl]]
        bot = rows[y1[sl]]
        bot -= top
        bot *= ty[sl, None]
        bot += top
        out[sl] = bot

    par_rows(h, work)
    return out


def fbm(h, w, base_cells, octaves, rng):
    """Fractal (multi-octave) tileable noise in roughly [0, 1]."""
    total = np.zeros((h, w), np.float32)
    amp, norm = 1.0, 0.0
    cells = max(1, int(base_cells))
    for _ in range(octaves):
        total += value_noise(h, w, cells, cells, rng) * amp
        norm += amp
        amp *= 0.5
        cells *= 2
    return total / norm


def smoothstep(e0, e1, x):
    if e1 < e0:
        return 1.0 - smoothstep(e1, e0, x)
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def hash01(a, b, seed, salt=0):
    """Deterministic per-cell random number in [0, 1) from two int arrays.
    For big per-pixel arrays with few distinct cells, the hash is computed once
    per cell and looked up, which gives exactly the same numbers much faster."""
    a, b = np.broadcast_arrays(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64))
    if a.size > 65536:
        amin, amax, bmin, bmax = int(a.min()), int(a.max()), int(b.min()), int(b.max())
        na, nb = amax - amin + 1, bmax - bmin + 1
        if na * nb <= (1 << 22):
            ta, tb = np.meshgrid(np.arange(amin, amax + 1), np.arange(bmin, bmax + 1), indexing="ij")
            table = _hash01_raw(ta, tb, seed, salt)
            return table[a - amin, b - bmin]
    return _hash01_raw(a, b, seed, salt)


def _hash01_raw(a, b, seed, salt):
    a = np.asarray(a, dtype=np.int64).astype(np.uint64)
    b = np.asarray(b, dtype=np.int64).astype(np.uint64)
    k = np.uint64((seed * 83492791 + salt * 2654435761 + 12345) & 0xFFFFFFFF)
    h = (a * np.uint64(73856093)) ^ (b * np.uint64(19349663)) ^ k
    h &= np.uint64(0xFFFFFFFF)
    h ^= h >> np.uint64(13)
    h *= np.uint64(0x5BD1E995)
    h &= np.uint64(0xFFFFFFFF)
    h ^= h >> np.uint64(15)
    return ((h & np.uint64(0xFFFFFF)).astype(np.float32)) / 16777216.0


def pixel_grid(size):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    return yy + 0.5, xx + 0.5


def rect_cells(size, cols, rows, offset=0.0, row_offsets=None):
    """Running-bond rectangular cells. Returns (dist_to_edge_px, id_a, id_b, fx, fy, bw, bh)."""
    yy, xx = pixel_grid(size)
    bw, bh = size / cols, size / rows
    v = yy / size * rows
    row = np.floor(v).astype(np.int64)
    fy = v - row
    shift = (row % 2) * offset
    if row_offsets is not None:
        shift = row_offsets[row % rows]
    u = xx / size * cols + shift
    col = np.floor(u).astype(np.int64)
    fx = u - col
    row %= rows
    col %= cols
    ex = np.minimum(fx, 1.0 - fx) * bw
    ey = np.minimum(fy, 1.0 - fy) * bh
    return np.minimum(ex, ey), col, row, fx, fy, bw, bh


def hex_cells(size, cols, rows):
    """Pointy-top hexagon cells. Returns (dist_to_edge_px, id_a, id_b, cell_px)."""
    yy, xx = pixel_grid(size)
    cw, ch = size / cols, size / rows
    qx = xx / cw
    qy = yy / ch * SQRT3
    sx, sy = 1.0, SQRT3
    ax = np.mod(qx, sx) - sx / 2
    ay = np.mod(qy, sy) - sy / 2
    bx = np.mod(qx - sx / 2, sx) - sx / 2
    by = np.mod(qy - sy / 2, sy) - sy / 2
    use_a = (ax * ax + ay * ay) < (bx * bx + by * by)
    lx = np.where(use_a, ax, bx)
    ly = np.where(use_a, ay, by)
    ia = np.floor(qx / sx).astype(np.int64) % cols
    ja = np.floor(qy / sy).astype(np.int64) % rows
    ib = np.floor((qx - sx / 2) / sx).astype(np.int64) % cols
    jb = np.floor((qy - sy / 2) / sy).astype(np.int64) % rows
    id_a = np.where(use_a, ia * 2, ib * 2 + 1)
    id_b = np.where(use_a, ja, jb)
    k = SQRT3 / 2
    edge = 0.5 - np.maximum(np.abs(lx), np.maximum(np.abs(0.5 * lx + k * ly), np.abs(0.5 * lx - k * ly)))
    return edge * cw, id_a, id_b, cw


def voronoi(size, n, rng, jitter=0.85, return_f1=False):
    """Tileable Voronoi. Returns (edge_dist_in_cells, cell_id[, dist_to_own_point])."""
    n = max(1, int(n))
    g = (np.arange(size, dtype=np.float32) + 0.5) * n / size
    c0 = np.floor(g).astype(np.int64)
    jx = (0.5 + (rng.random((n, n)) - 0.5) * jitter).astype(np.float32)
    jy = (0.5 + (rng.random((n, n)) - 0.5) * jitter).astype(np.float32)
    # Track *squared* distances and take the square root once at the end
    # (sqrt keeps order, so nearest / second-nearest are the same points).
    f1 = np.full((size, size), np.inf, np.float64)
    f2 = np.full((size, size), np.inf, np.float64)
    cid = np.zeros((size, size), np.int64)
    for oy in (-1, 0, 1):
        cy = c0 + oy
        iy = cy % n
        for ox in (-1, 0, 1):
            cx = c0 + ox
            ix = cx % n
            dx = g[None, :] - (cx[None, :] + jx[np.ix_(iy, ix)])
            dy = g[:, None] - (cy[:, None] + jy[np.ix_(iy, ix)])
            dx *= dx
            dy *= dy
            dx += dy
            d = dx
            closer = d < f1
            np.minimum(f2, d, out=f2)
            np.copyto(f2, f1, where=closer)
            np.copyto(f1, d, where=closer)
            np.copyto(cid, iy[:, None] * n + ix[None, :], where=closer)
    np.sqrt(f1, out=f1)
    np.sqrt(f2, out=f2)
    if return_f1:
        return (f2 - f1) * 0.5, cid, f1
    return (f2 - f1) * 0.5, cid


# ---------------------------------------------------------------------------
# Procedural texture patterns
# ---------------------------------------------------------------------------

def _color(v):
    return np.array(v, np.float32)[None, None, :]


def round_spots(size, density, rng, radius=2.5):
    """Tileable soft round spots (pits, specks) in [0, 1]."""
    seeds = (rng.random((size, size)) < density).astype(np.float32)
    if not seeds.any():
        return np.zeros((size, size), np.float32)
    s = gauss_blur(seeds, radius * size / 1024.0 * 2)
    pk = float(s.max())
    return smoothstep(0.25 * pk, 0.6 * pk, s)


def shade_cells(p, size, d, id_a, id_b, unit, rng, two_tone=False):
    """Shared look for anything made of pieces separated by gaps
    (bricks, tiles, stones, plates...)."""
    cells_across = max(2, int(round(size / unit)))
    edge_noise = fbm(size, size, cells_across * 2, 4, rng) - 0.5
    d = d - p.tex_gap * unit * 0.5 + edge_noise * p.tex_wear * unit * 0.25

    bevel_px = max(p.tex_bevel * unit * 0.5, 0.5)
    mask = smoothstep(0.0, bevel_px, d)
    solid = (d > 0).astype(np.float32)

    r1 = hash01(id_a, id_b, p.tex_seed, 1)
    r2 = hash01(id_a, id_b, p.tex_seed, 2)
    r3 = hash01(id_a, id_b, p.tex_seed, 3)

    fine = fbm(size, size, max(8, size // 64), 5, rng)
    coarse = fbm(size, size, cells_across, 3, rng)
    grit = fbm(size, size, max(16, size // 16), 3, rng)  # small-scale surface texture

    # Base sits at ~0.6 so bumps have room above it before hitting 1.0
    piece_h = (0.58 + 0.08 * r3
               + (normalize01(grit) - 0.5) * 0.25 * p.tex_bump
               + (normalize01(fine) - 0.5) * 0.25 * p.tex_bump
               + (normalize01(coarse) - 0.5) * 0.15 * p.tex_bump)
    gap_h = 0.1 + (fine - 0.5) * 0.1
    height = np.clip(gap_h * (1 - mask) + piece_h * mask, 0.0, 1.0)

    c1, c2 = _color(p.tex_color1), _color(p.tex_color2)
    pick = (r1 > 0.5).astype(np.float32) if two_tone else r1
    piece_col = c1 + (c2 - c1) * pick[..., None]
    piece_col = piece_col * (1.0 + (r2[..., None] - 0.5) * 2.0 * p.tex_variation)
    grime = min(1.0, 0.15 + p.tex_bump)  # smooth glossy tiles get less mottling
    piece_col = piece_col * (1.0 + (fine[..., None] - 0.5) * 0.3 * grime)
    gap_col = _color(p.tex_gap_color) * (0.8 + 0.4 * fine[..., None])
    cmask = np.clip(solid * 0.5 + mask * 0.5, 0, 1)[..., None]
    albedo = gap_col * (1 - cmask) + piece_col * cmask

    rough = (p.tex_rough_gap * (1 - mask)
             + (p.tex_rough + (fine - 0.5) * 0.15) * mask)
    return {"height": height, "albedo": albedo, "rough": rough, "mask": mask,
            "fine": fine, "r1": r1, "r2": r2, "r3": r3}


def pat_bricks(p, size, rng):
    d, a, b, fx, fy, bw, bh = rect_cells(size, p.tex_count_x, p.tex_count_y, p.tex_offset)
    return shade_cells(p, size, d, a, b, min(bw, bh), rng)


def pat_tiles(p, size, rng):
    d, a, b, fx, fy, bw, bh = rect_cells(size, p.tex_count_x, p.tex_count_y, 0.0)
    return shade_cells(p, size, d, a, b, min(bw, bh), rng)


def pat_hex(p, size, rng):
    d, a, b, unit = hex_cells(size, p.tex_count_x, p.tex_count_y)
    return shade_cells(p, size, d, a, b, unit, rng, two_tone=True)


def pat_stones(p, size, rng):
    edge, cid = voronoi(size, p.tex_count_x, rng)
    unit = size / p.tex_count_x
    return shade_cells(p, size, edge * unit, cid, cid * 7 + 3, unit, rng)


def pat_planks(p, size, rng):
    rows, cols = p.tex_count_y, p.tex_count_x
    row_offsets = rng.random(rows).astype(np.float32)
    d, a, b, fx, fy, bw, bh = rect_cells(size, cols, rows, row_offsets=row_offsets)
    out = shade_cells(p, size, d, a, b, min(bw, bh), rng)

    # Wood grain: long stretched noise bands that run along each plank.
    warp = value_noise(size, size, rows * 6, max(1, cols * 2), rng)
    streak = value_noise(size, size, rows * 40, max(2, cols * 3), rng)
    yy, _ = pixel_grid(size)
    local_v = (yy / size * rows) % 1.0
    phase = local_v * (3.0 + 4.0 * out["r3"]) + warp * 3.0 + out["r1"] * 10.0
    grain = 0.5 + 0.5 * np.sin(phase * 2 * np.pi)
    grain = np.clip(grain * 0.6 + streak * 0.4, 0, 1)
    g = grain * p.tex_detail

    c1, c2 = _color(p.tex_color1), _color(p.tex_color2)
    wood = c1 + (c2 - c1) * np.clip(g[..., None] + out["r1"][..., None] * 0.3, 0, 1)
    wood = wood * (1.0 + (out["r2"][..., None] - 0.5) * 2.0 * p.tex_variation)
    m = out["mask"][..., None]
    out["albedo"] = out["albedo"] * (1 - m) + wood * m
    out["height"] = out["height"] - g * 0.04 * out["mask"]
    out["rough"] = out["rough"] + (g - 0.5) * 0.1 * out["mask"]
    return out


def pat_metal(p, size, rng):
    d, a, b, fx, fy, bw, bh = rect_cells(size, p.tex_count_x, p.tex_count_y, 0.0)
    unit = min(bw, bh)
    out = shade_cells(p, size, d, a, b, unit, rng)

    # Brushed streaks / scratches
    lines_v = value_noise(size, size, 3, max(8, size // 2), rng)
    lines_h = value_noise(size, size, max(8, size // 2), 3, rng)
    brushed = value_noise(size, size, 6, max(8, size // 4), rng)
    scratch = np.maximum(smoothstep(0.86, 0.98, lines_v), smoothstep(0.9, 0.99, lines_h)) * p.tex_detail
    out["height"] = out["height"] - scratch * 0.02 + (brushed - 0.5) * 0.01
    out["rough"] = np.clip(out["rough"] + (brushed - 0.5) * 0.15 + scratch * 0.25, 0, 1)
    out["albedo"] = out["albedo"] * (1.0 + scratch[..., None] * 0.15)

    if p.tex_rivets:
        inset = 0.07 * unit
        radius = 0.028 * unit
        dx = np.minimum(fx, 1 - fx) * bw - inset
        dy = np.minimum(fy, 1 - fy) * bh - inset
        dist = np.sqrt(dx * dx + dy * dy)
        dome = np.sqrt(np.clip(1.0 - (dist / radius) ** 2, 0, 1))
        out["height"] = np.maximum(out["height"], out["height"] * (1 - (dome > 0)) + (0.8 + 0.12 * dome) * (dome > 0))
        out["rough"] = out["rough"] * (1 - (dome > 0) * 0.3)

    out["metal"] = np.clip(0.4 + 0.6 * out["mask"], 0, 1)
    return out


def pat_concrete(p, size, rng):
    scale = max(1, p.tex_count_x)
    base = fbm(size, size, scale, 6, rng)
    big = fbm(size, size, max(1, scale // 2), 3, rng)
    pits = np.maximum(round_spots(size, 0.0006, rng, 2.0), round_spots(size, 0.00015, rng, 4.0)) * p.tex_detail
    edge, _ = voronoi(size, max(2, scale), rng, jitter=1.0)
    crack_mask = smoothstep(0.45, 0.65, fbm(size, size, scale * 2, 3, rng))
    crack = smoothstep(0.03, 0.0, edge) * crack_mask * p.tex_detail

    height = 0.6 + (base - 0.5) * 0.35 * p.tex_bump - pits * 0.3 - crack * 0.4
    c1, c2 = _color(p.tex_color1), _color(p.tex_color2)
    albedo = c1 + (c2 - c1) * np.clip(big * 1.5 - 0.25, 0, 1)[..., None]
    albedo = albedo * (0.9 + 0.2 * base[..., None]) * (1 - 0.35 * (pits + crack))[..., None]
    rough = np.clip(p.tex_rough + (base - 0.5) * 0.1, 0, 1)
    return {"height": np.clip(height, 0, 1), "albedo": albedo, "rough": rough}


def pat_rock(p, size, rng):
    scale = max(1, p.tex_count_x)
    n1 = fbm(size, size, scale, 6, rng)
    ridged = 1.0 - np.abs(2.0 * n1 - 1.0)
    n2 = fbm(size, size, scale * 4, 4, rng)
    height = normalize01(ridged ** 2 * 0.65 + n2 * 0.35)
    height = 0.5 + (height - 0.5) * min(1.0, p.tex_bump)
    speck = round_spots(size, 0.002, rng, 1.5) * p.tex_detail
    c1, c2 = _color(p.tex_color1), _color(p.tex_color2)
    t = np.clip(height + (n2 - 0.5) * 0.4, 0, 1)[..., None]
    albedo = (c1 + (c2 - c1) * t) * (1 - 0.3 * speck[..., None])
    rough = np.clip(p.tex_rough + (0.5 - height) * 0.15, 0, 1)
    return {"height": height, "albedo": albedo, "rough": rough}


def pat_fabric(p, size, rng):
    nx, ny = max(1, p.tex_count_x), max(1, p.tex_count_y)
    yy, xx = pixel_grid(size)
    u = xx / size * nx
    v = yy / size * ny
    i = np.floor(u).astype(np.int64)
    j = np.floor(v).astype(np.int64)
    fu, fv = u - i, v - j
    warp_top = ((i + j) % 2) == 0

    prof_u = np.sin(np.pi * fu) ** 0.6
    prof_v = np.sin(np.pi * fv) ** 0.6
    und_u = 0.6 + 0.4 * np.sin(np.pi * fu)
    und_v = 0.6 + 0.4 * np.sin(np.pi * fv)
    fib_warp = value_noise(size, size, ny * 2, nx * 12, rng)
    fib_weft = value_noise(size, size, ny * 12, nx * 2, rng)
    h_warp = prof_u * und_v + (fib_warp - 0.5) * 0.2 * p.tex_detail
    h_weft = prof_v * und_u + (fib_weft - 0.5) * 0.2 * p.tex_detail
    height = np.where(warp_top, h_warp, h_weft)
    height = 0.5 + (height - 0.5) * min(1.0, p.tex_bump * 2)

    var_i = 1.0 + (hash01(i % nx, 0, p.tex_seed, 5) - 0.5) * 2 * p.tex_variation
    var_j = 1.0 + (hash01(0, j % ny, p.tex_seed, 6) - 0.5) * 2 * p.tex_variation
    c1, c2 = _color(p.tex_color1), _color(p.tex_color2)
    col = np.where(warp_top[..., None], c1 * var_i[..., None], c2 * var_j[..., None])
    albedo = col * (0.65 + 0.35 * np.clip(height, 0, 1))[..., None]
    rough = np.full((size, size), p.tex_rough, np.float32)
    return {"height": np.clip(height, 0, 1), "albedo": albedo, "rough": rough}


def _mix(a, b, t):
    return a + (b - a) * t


def _frac(a):
    return a - np.floor(a)


def _wood_grain(out, p, across, warp):
    """Add wood grain bands that run along each plank (used by planks/herringbone)."""
    phase = across * (2.0 + 3.0 * out["r3"]) + warp * 3.0 + out["r1"] * 10.0
    grain = 0.5 + 0.5 * np.sin(phase * 2 * np.pi)
    g = grain * p.tex_detail
    c1, c2 = _color(p.tex_color1), _color(p.tex_color2)
    wood = c1 + (c2 - c1) * np.clip(g[..., None] + out["r1"][..., None] * 0.3, 0, 1)
    wood = wood * (1.0 + (out["r2"][..., None] - 0.5) * 2.0 * p.tex_variation)
    m = out["mask"][..., None]
    out["albedo"] = out["albedo"] * (1 - m) + wood * m
    out["height"] = out["height"] - g * 0.04 * out["mask"]
    out["rough"] = out["rough"] + (g - 0.5) * 0.1 * out["mask"]
    return out


def pat_roof(p, size, rng):
    """Overlapping clay / slate roof tiles with rounded bottom edges."""
    rows = max(2, p.tex_count_y)
    cols = max(1, p.tex_count_x)
    yy, xx = pixel_grid(size)
    bw, bh = size / cols, size / rows
    v = yy / size * rows
    row = np.floor(v).astype(np.int64)
    fy = v - row

    def tile_x(r):
        u = xx / size * cols + (r % 2) * 0.5
        c = np.floor(u).astype(np.int64)
        return c, u - c

    _, fx0 = tile_x(row)
    wobble = (fbm(size, size, cols * 4, 3, rng) - 0.5) * p.tex_wear * 0.12
    depth = 0.45 * p.tex_bevel
    edge = depth * (1.0 - np.sqrt(np.clip(1.0 - (2 * fx0 - 1) ** 2, 0, 1))) + wobble
    under = fy < edge  # below this row's curved bottom edge we see the tile underneath
    r_eff = np.where(under, row - 1, row)
    fy_eff = np.where(under, fy + 1.0, fy)
    col_eff, fx_eff = tile_x(r_eff)

    fine = fbm(size, size, max(8, size // 64), 5, rng)
    height = 0.9 - 0.32 * fy_eff / (1.0 + depth)
    height = height + np.sin(np.pi * fx_eff) * 0.12 * p.tex_detail  # barrel curve
    height = height + (fine - 0.5) * 0.08 * p.tex_bump
    ex = np.minimum(fx_eff, 1 - fx_eff) * bw
    gap_px = p.tex_gap * bw * 0.5
    gm = smoothstep(gap_px, gap_px + max(1.0, bw * 0.02), ex)
    height = height * gm + 0.35 * (1 - gm)

    ida, idb = col_eff % cols, r_eff % rows
    r1 = hash01(ida, idb, p.tex_seed, 1)
    r2 = hash01(ida, idb, p.tex_seed, 2)
    c1, c2 = _color(p.tex_color1), _color(p.tex_color2)
    tile = (c1 + (c2 - c1) * r1[..., None]) * (1 + (r2[..., None] - 0.5) * 2 * p.tex_variation)
    tile = tile * (0.85 + 0.3 * fine[..., None]) * (0.8 + 0.2 * (1 - fy_eff / 1.5))[..., None]
    gap_col = _color(p.tex_gap_color)
    albedo = tile * gm[..., None] + gap_col * (1 - gm[..., None])
    rough = p.tex_rough * gm + p.tex_rough_gap * (1 - gm) + (fine - 0.5) * 0.1
    return {"height": height, "albedo": albedo, "rough": rough}


def pat_herringbone(p, size, rng):
    """Herringbone parquet / pavers. Plank length = Length x width."""
    N = max(2, p.tex_count_y)
    m = max(1, int(round(p.tex_count_x / (2.0 * N))))
    S = m * 2 * N  # plank widths across the texture (keeps it seamless)
    yy, xx = pixel_grid(size)
    x = xx / size * S
    y = yy / size * S
    w_px = size / S

    found = np.zeros((size, size), bool)
    kind = np.zeros((size, size), np.int64)
    kk = np.zeros((size, size), np.int64)
    bb = np.zeros((size, size), np.int64)
    along = np.zeros((size, size), np.float32)
    across = np.zeros((size, size), np.float32)
    b0 = np.floor((x - y) / (2 * N)).astype(np.int64)
    for db in (-1, 0, 1, 2):
        b = b0 + db
        xp = x - b * N
        yp = y + b * N
        kH = np.floor(yp).astype(np.int64)
        selH = (xp - kH >= 0) & (xp - kH < N) & ~found
        kind[selH] = 0
        kk[selH] = kH[selH]
        bb[selH] = b[selH]
        along[selH] = ((xp - kH) / N)[selH]
        across[selH] = (yp - kH)[selH]
        found |= selH
        kV = np.floor(xp).astype(np.int64)
        selV = (yp - kV >= 1) & (yp - kV < 1 + N) & ~found
        kind[selV] = 1
        kk[selV] = kV[selV]
        bb[selV] = b[selV]
        along[selV] = ((yp - kV - 1) / N)[selV]
        across[selV] = (xp - kV)[selV]
        found |= selV

    # Same physical plank on both sides of the texture edge -> same random id.
    bmod = np.mod(bb, 2 * m)
    shift = bmod >= m
    bmod = np.where(shift, bmod - m, bmod)
    kmod = np.mod(np.where(shift, kk - m * N, kk), S)
    id_a = bmod * 2 + kind
    id_b = kmod

    ex = np.minimum(along, 1 - along) * N * w_px
    ey = np.minimum(across, 1 - across) * w_px
    out = shade_cells(p, size, np.minimum(ex, ey), id_a, id_b, w_px, rng)
    noise_h = value_noise(size, size, S * 3, max(1, S // N), rng)  # grain along x
    noise_v = value_noise(size, size, max(1, S // N), S * 3, rng)  # grain along y
    return _wood_grain(out, p, across, np.where(kind == 0, noise_h, noise_v))


def pat_corrugated(p, size, rng):
    """Corrugated iron sheets with rust."""
    ridges = max(1, p.tex_count_x)
    sheets = max(1, p.tex_count_y)
    yy, xx = pixel_grid(size)
    wave = 0.5 + 0.5 * np.cos(2 * np.pi * xx / size * ridges)
    v = yy / size * sheets
    sheet = np.floor(v).astype(np.int64) % sheets
    fy = v - np.floor(v)
    fine = fbm(size, size, max(8, size // 64), 4, rng)
    dents = fbm(size, size, 3, 3, rng)
    height = 0.3 + 0.45 * wave + 0.1 * fy + (dents - 0.5) * 0.2 * p.tex_bump

    streaks = value_noise(size, size, 2, ridges * 2, rng)
    rust_n = fbm(size, size, 4, 5, rng)
    level = 1.0 - p.tex_detail * 0.9
    rust = smoothstep(level - 0.08, level + 0.08, rust_n * 0.6 + streaks * 0.3 + (1 - wave) * 0.1 + (1 - fy) * 0.05)
    rust = rust * (p.tex_detail > 0)
    height = height - rust * 0.02 + (fine - 0.5) * 0.04 * rust

    sv = hash01(sheet, sheet * 3, p.tex_seed, 7)
    metal_col = _color(p.tex_color1) * (0.85 + 0.2 * fine[..., None]) * (1 + (sv[..., None] - 0.5) * 2 * p.tex_variation)
    rust_col = _color(p.tex_color2) * (0.6 + 0.7 * fine[..., None])
    albedo = _mix(metal_col, rust_col, rust[..., None])
    rough = _mix(p.tex_rough + (dents - 0.5) * 0.1, 0.9, rust)
    return {"height": height, "albedo": albedo, "rough": rough, "metal": 1.0 - rust * 0.95}


def pat_rusty(p, size, rng):
    """Painted metal with chipped paint, bare metal edges and rust."""
    base = fbm(size, size, max(2, p.tex_count_x), 6, rng)
    level = 1.0 - p.tex_detail
    paint = 1.0 - smoothstep(level - 0.1, level - 0.05, base)
    rust = smoothstep(level - 0.02, level + 0.06, base)
    bare = np.clip(1.0 - paint - rust, 0, 1)
    fine = fbm(size, size, max(8, size // 32), 4, rng)
    grain = fbm(size, size, max(16, size // 16), 3, rng)

    height = 0.5 + 0.1 * paint + (fine - 0.5) * 0.02 + (grain - 0.5) * 0.12 * rust * p.tex_bump
    paint_col = _color(p.tex_color1) * (1 + (fine[..., None] - 0.5) * 0.3 * p.tex_variation * 2)
    bare_col = _color((0.62, 0.62, 0.64)) * (0.9 + 0.2 * grain[..., None])
    rust_col = _color(p.tex_color2) * (0.55 + 0.8 * grain[..., None])
    albedo = paint_col * paint[..., None] + bare_col * bare[..., None] + rust_col * rust[..., None]
    rough = p.tex_rough * paint + 0.3 * bare + 0.9 * rust
    return {"height": height, "albedo": albedo, "rough": rough, "metal": bare}


def pat_bark(p, size, rng):
    """Tree bark: long vertical ridges and furrows."""
    cx, cy = max(1, p.tex_count_x), max(1, p.tex_count_y)
    n = np.zeros((size, size), np.float32)
    amp, norm = 1.0, 0.0
    for o in range(5):
        n += value_noise(size, size, cy * 2 ** o, cx * 2 ** o, rng) * amp
        norm += amp
        amp *= 0.55
    n /= norm
    ridged = 1.0 - np.abs(2.0 * normalize01(n) - 1.0)
    plates = smoothstep(0.25, 0.9, ridged)  # flat-topped ridges with sharp, deep furrows
    breaks = smoothstep(0.75, 0.85, value_noise(size, size, cy * 5, cx * 2, rng)) * p.tex_detail
    fiber = fbm(size, size, max(8, cx * 4), 3, rng) * 0.5 + value_noise(size, size, cy * 12, cx * 24, rng) * 0.5
    height = plates * 0.75 + (fiber - 0.5) * 0.3 * p.tex_bump - breaks * 0.35 * plates
    big = fbm(size, size, 2, 3, rng)
    c1, c2 = _color(p.tex_color1), _color(p.tex_color2)
    albedo = _mix(c2, c1, np.clip(plates * 1.2, 0, 1)[..., None])
    albedo = albedo * (1 + (big[..., None] - 0.5) * 2 * p.tex_variation) * (0.85 + 0.3 * fiber[..., None])
    rough = np.clip(p.tex_rough + (1 - plates) * 0.05, 0, 1)
    return {"height": np.clip(height, 0, 1), "albedo": albedo, "rough": rough}


def pat_gravel(p, size, rng):
    """Packed rounded pebbles with dirt between."""
    edge, cid, f1 = voronoi(size, p.tex_count_x, rng, jitter=1.0, return_f1=True)
    r1 = hash01(cid, cid * 7 + 1, p.tex_seed, 1)
    r2 = hash01(cid, cid * 7 + 1, p.tex_seed, 2)
    r3 = hash01(cid, cid * 7 + 1, p.tex_seed, 3)
    # Each pebble is a round blob of random size around its cell's point,
    # trimmed where it would touch a neighbour.
    radius = (0.5 + 0.2 * r3) * (1.0 - p.tex_gap * 0.6)
    s_round = np.clip((radius - f1) / (radius * (0.25 + 0.75 * p.tex_bevel)), 0, 1)
    s_edge = np.clip((edge - p.tex_gap * 0.12) / 0.08, 0, 1)
    s = np.minimum(s_round, s_edge)
    dome = np.sqrt(1 - (1 - s) ** 2)
    stone = (s > 0).astype(np.float32)
    fine = fbm(size, size, max(8, size // 32), 4, rng)
    dirt = fbm(size, size, max(4, p.tex_count_x), 3, rng)
    height = np.where(stone > 0, 0.2 + dome * (0.55 + 0.35 * r3), 0.12 + dirt * 0.1) + (fine - 0.5) * 0.05 * p.tex_bump
    c1, c2 = _color(p.tex_color1), _color(p.tex_color2)
    sc = (c1 + (c2 - c1) * r1[..., None]) * (1 + (r2[..., None] - 0.5) * 2 * p.tex_variation)
    sc = sc * (0.85 + 0.3 * fine[..., None])
    dc = _color(p.tex_gap_color) * (0.75 + 0.5 * dirt[..., None])
    m = smoothstep(0, 0.15, s)[..., None]
    albedo = dc * (1 - m) + sc * m
    rough = p.tex_rough * m[..., 0] + p.tex_rough_gap * (1 - m[..., 0])
    return {"height": np.clip(height, 0, 1), "albedo": albedo, "rough": rough}


def pat_sand(p, size, rng):
    """Wind-blown sand ripples with fine grain."""
    waves = max(1, p.tex_count_x)
    yy, xx = pixel_grid(size)
    warp = fbm(size, size, 3, 4, rng)
    phase = yy / size * waves + (warp - 0.5) * 2.5
    f = _frac(phase)
    prof = np.where(f < 0.75, f / 0.75, (1.0 - f) / 0.25)
    prof = prof * prof * (3 - 2 * prof)
    grain = gauss_blur(rng.random((size, size)).astype(np.float32), 1)
    grain = normalize01(grain)
    patch = fbm(size, size, 2, 3, rng)
    height = 0.5 + (prof - 0.5) * 0.5 * p.tex_bump + (grain - 0.5) * 0.2 * p.tex_detail
    c1, c2 = _color(p.tex_color1), _color(p.tex_color2)
    albedo = _mix(c1, c2, np.clip((1 - prof) * 0.5 + (patch - 0.5) * p.tex_variation * 2, 0, 1)[..., None])
    albedo = albedo * (0.9 + 0.2 * grain[..., None])
    rough = np.full((size, size), p.tex_rough, np.float32) - grain * 0.05
    return {"height": np.clip(height, 0, 1), "albedo": albedo, "rough": rough}


def pat_mud(p, size, rng):
    """Dry cracked mud: plates that curl up at the edges, with deep cracks."""
    edge, cid = voronoi(size, p.tex_count_x, rng, jitter=1.0)
    unit = size / p.tex_count_x
    epx = edge * unit
    wob = (fbm(size, size, p.tex_count_x * 3, 3, rng) - 0.5) * p.tex_wear * unit * 0.12
    cw = max(1.0, p.tex_gap * unit * 0.5)
    crack = 1.0 - smoothstep(cw * 0.4, cw, epx + wob)
    edge2, _ = voronoi(size, p.tex_count_x * 3, rng, jitter=1.0)
    small = (1.0 - smoothstep(0.0, 0.04, edge2)) * p.tex_detail * 0.6
    curl = 1.0 - smoothstep(0, unit * (0.1 + 0.3 * p.tex_bevel), epx)
    fine = fbm(size, size, max(8, size // 32), 4, rng)
    r1 = hash01(cid, cid * 5 + 2, p.tex_seed, 1)
    plate_h = 0.6 + 0.15 * curl + (fine - 0.5) * 0.1 * p.tex_bump + (r1 - 0.5) * 0.05
    height = plate_h * (1 - crack) + 0.08 * crack - small * 0.15
    c1, c2 = _color(p.tex_color1), _color(p.tex_color2)
    plate = _mix(c1, c2, r1[..., None] * p.tex_variation * 2) * (0.85 + 0.3 * fine[..., None])
    albedo = _mix(plate, _color(p.tex_gap_color), np.clip(crack + small * 0.6, 0, 1)[..., None])
    rough = p.tex_rough * (1 - crack) + p.tex_rough_gap * crack
    return {"height": np.clip(height, 0, 1), "albedo": albedo, "rough": rough}


def pat_asphalt(p, size, rng):
    """Road asphalt: dark binder with light stone aggregate and cracks."""
    fine = fbm(size, size, max(16, size // 16), 3, rng)
    s1 = round_spots(size, 0.006 * p.tex_detail, rng, 0.8)
    s2 = round_spots(size, 0.0012 * p.tex_detail, rng, 1.8)
    stones = np.maximum(s1, s2)
    tone = value_noise(size, size, size // 6, size // 6, rng)
    pits = round_spots(size, 0.0008, rng, 1.2)
    edge, _ = voronoi(size, max(1, p.tex_count_x), rng, jitter=1.0)
    crack = smoothstep(0.025, 0.0, edge) * smoothstep(0.45, 0.65, fbm(size, size, max(2, p.tex_count_x * 2), 3, rng))
    crack = crack * min(1.0, p.tex_wear)
    height = 0.45 + (fine - 0.5) * 0.1 * p.tex_bump + stones * 0.12 - pits * 0.15 - crack * 0.3
    base = _color(p.tex_color1) * (0.8 + 0.4 * fine[..., None])
    stone_col = _color(p.tex_color2) * (0.6 + 0.6 * tone[..., None])
    albedo = _mix(base, stone_col, stones[..., None]) * (1 - 0.5 * np.maximum(crack, pits))[..., None]
    rough = np.clip(p.tex_rough - stones * 0.15, 0, 1)
    return {"height": np.clip(height, 0, 1), "albedo": albedo, "rough": rough}


def pat_plaster(p, size, rng):
    """Hand-troweled plaster / stucco wall."""
    sc = max(1, p.tex_count_x)
    big = fbm(size, size, max(1, sc // 2), 3, rng)
    # Trowel marks: soft strokes in two directions, blended patch by patch,
    # plus a finer layer so they read as hand-applied rather than as stripes.
    sx = fbm(size, size, sc * 3, 3, rng) * 0.5 + value_noise(size, size, sc * 3, sc * 2, rng) * 0.5
    sy = fbm(size, size, sc * 3, 3, rng) * 0.5 + value_noise(size, size, sc * 2, sc * 3, rng) * 0.5
    sel = smoothstep(0.3, 0.7, value_noise(size, size, sc * 2, sc * 2, rng))
    trowel = gauss_blur(sx * sel + sy * (1 - sel), max(1, size // 256))
    stucco = fbm(size, size, sc * 8, 3, rng)
    pits = round_spots(size, 0.0004, rng, 1.5) * p.tex_wear
    height = 0.5 + (trowel - 0.5) * 0.35 * p.tex_bump + (stucco - 0.5) * 0.25 * p.tex_detail - pits * 0.2
    c1, c2 = _color(p.tex_color1), _color(p.tex_color2)
    albedo = _mix(c1, c2, np.clip((big - 0.5) * 2 * p.tex_variation * 2 + 0.5, 0, 1)[..., None])
    albedo = albedo * (0.92 + 0.16 * trowel[..., None]) * (1 - 0.3 * pits[..., None])
    rough = np.clip(p.tex_rough + (trowel - 0.5) * 0.15, 0, 1)
    return {"height": np.clip(height, 0, 1), "albedo": albedo, "rough": rough}


def pat_terrazzo(p, size, rng):
    """Polished terrazzo: stone chips set in a smooth matrix."""
    c1, c2, c3 = _color(p.tex_color1), _color(p.tex_color2), _color(p.tex_gap_color)
    fine = fbm(size, size, max(8, size // 32), 4, rng)
    albedo = c1 * (0.95 + 0.1 * fine[..., None])
    chips_all = np.zeros((size, size), np.float32)
    for scale, salt in ((max(1, p.tex_count_x), 11), (max(1, p.tex_count_x) * 3, 12)):
        edge, cid = voronoi(size, scale, rng, jitter=1.0)
        h = hash01(cid, cid + salt, p.tex_seed, salt)
        h2 = hash01(cid, cid + salt, p.tex_seed, salt + 20)
        sel = (h < 0.3 + 0.5 * p.tex_detail).astype(np.float32)
        chip = smoothstep(p.tex_gap * 0.3, p.tex_gap * 0.3 + 0.02, edge) * sel
        col = np.where((h2 < 0.4)[..., None], c2, np.where((h2 < 0.75)[..., None], c3, (c2 + c3) * 0.5 + 0.15))
        col = col * (1 + (hash01(cid, cid, p.tex_seed, salt + 40)[..., None] - 0.5) * 2 * p.tex_variation)
        albedo = _mix(albedo, col, chip[..., None])
        chips_all = np.maximum(chips_all, chip)
    height = 0.5 + chips_all * 0.01 + (fine - 0.5) * 0.02 * p.tex_bump
    rough = np.clip(p.tex_rough - chips_all * 0.05, 0, 1)
    return {"height": height, "albedo": albedo, "rough": rough}


def pat_leather(p, size, rng):
    """Pebbled leather grain with soft creases."""
    edge, cid = voronoi(size, p.tex_count_x, rng, jitter=1.0)
    pebble = np.sqrt(smoothstep(0.0, 0.35, edge))
    wr = fbm(size, size, 3, 5, rng)
    crease = smoothstep(0.93, 1.0, 1.0 - np.abs(2 * wr - 1)) * p.tex_detail
    big = fbm(size, size, 2, 3, rng)
    height = 0.4 + pebble * 0.3 * min(1.5, p.tex_bump * 2) + (wr - 0.5) * 0.15 - crease * 0.2
    c1, c2 = _color(p.tex_color1), _color(p.tex_color2)
    albedo = _mix(c1, c2, np.clip(big * p.tex_variation * 2, 0, 1)[..., None])
    albedo = albedo * (0.75 + 0.25 * pebble[..., None]) * (1 - 0.3 * crease[..., None])
    rough = np.clip(p.tex_rough + (1 - pebble) * 0.1, 0, 1)
    return {"height": np.clip(height, 0, 1), "albedo": albedo, "rough": rough}


PATTERNS = {
    "BRICKS": ("Bricks", pat_bricks),
    "STONES": ("Cobblestones", pat_stones),
    "GRAVEL": ("Gravel", pat_gravel),
    "ROOF": ("Roof Tiles", pat_roof),
    "TILES": ("Tiles", pat_tiles),
    "HEX": ("Hex Tiles", pat_hex),
    "TERRAZZO": ("Terrazzo", pat_terrazzo),
    "PLANKS": ("Wood Planks", pat_planks),
    "HERRINGBONE": ("Herringbone", pat_herringbone),
    "CONCRETE": ("Concrete", pat_concrete),
    "PLASTER": ("Plaster", pat_plaster),
    "ASPHALT": ("Asphalt", pat_asphalt),
    "ROCK": ("Rock", pat_rock),
    "MUD": ("Cracked Mud", pat_mud),
    "SAND": ("Sand", pat_sand),
    "BARK": ("Bark", pat_bark),
    "METAL": ("Metal Plates", pat_metal),
    "CORRUGATED": ("Corrugated Metal", pat_corrugated),
    "RUSTY": ("Rusty Metal", pat_rusty),
    "FABRIC": ("Fabric", pat_fabric),
    "LEATHER": ("Leather", pat_leather),
}

# Dropdown layout: (heading, [pattern keys]) with a tooltip per pattern.
PATTERN_GROUPS = (
    ("Masonry & Roofs", ["BRICKS", "STONES", "GRAVEL", "ROOF"]),
    ("Floors & Tiles", ["TILES", "HEX", "TERRAZZO", "PLANKS", "HERRINGBONE"]),
    ("Walls & Ground", ["CONCRETE", "PLASTER", "ASPHALT", "ROCK", "MUD", "SAND", "BARK"]),
    ("Metal & Fabric", ["METAL", "CORRUGATED", "RUSTY", "FABRIC", "LEATHER"]),
)
PATTERN_TIPS = {
    "BRICKS": "Brick wall with mortar",
    "STONES": "Irregular rounded cobblestones",
    "GRAVEL": "Packed rounded pebbles with dirt between them",
    "ROOF": "Overlapping clay or slate roof tiles",
    "TILES": "Square ceramic tiles with grout",
    "HEX": "Hexagon tiles in two colors",
    "TERRAZZO": "Polished floor with colored stone chips",
    "PLANKS": "Wooden floorboards with grain",
    "HERRINGBONE": "Zig-zag herringbone parquet or pavers",
    "CONCRETE": "Concrete with pits and cracks",
    "PLASTER": "Hand-troweled plaster or stucco wall",
    "ASPHALT": "Road asphalt with stone aggregate and cracks",
    "ROCK": "Rough natural rock surface",
    "MUD": "Dry cracked mud or earth",
    "SAND": "Wind-blown sand ripples",
    "BARK": "Tree bark with long ridges and furrows",
    "METAL": "Metal panels with seams, scratches and rivets",
    "CORRUGATED": "Corrugated iron sheets with rust",
    "RUSTY": "Painted metal with chipped paint and rust",
    "FABRIC": "Woven cloth",
    "LEATHER": "Pebbled leather grain with creases",
}


def _pattern_enum_items():
    items, i = [], 0
    for heading, keys in PATTERN_GROUPS:
        items.append(("", heading, ""))
        for k in keys:
            items.append((k, PATTERNS[k][0], PATTERN_TIPS[k], i))
            i += 1
    return items


# Which settings each pattern uses, and labels for the reused ones.
_PIECES = ["count_x", "count_y", "gap", "bevel", "wear", "bump",
           "color1", "color2", "gap_color", "variation", "rough", "rough_gap"]
PATTERN_UI = {
    "BRICKS": (_PIECES[:2] + ["offset"] + _PIECES[2:], {"count_x": "Columns", "count_y": "Rows", "gap": "Mortar Size", "gap_color": "Mortar Color"}),
    "TILES": (_PIECES, {"count_x": "Columns", "count_y": "Rows", "gap": "Grout Size", "gap_color": "Grout Color"}),
    "HEX": (_PIECES, {"count_x": "Columns", "count_y": "Row Pairs", "gap": "Grout Size", "gap_color": "Grout Color"}),
    "PLANKS": (_PIECES + ["detail"], {"count_x": "Planks per Row", "count_y": "Rows", "gap": "Gap Size", "detail": "Wood Grain", "color1": "Light Wood", "color2": "Dark Wood"}),
    "STONES": ([k for k in _PIECES if k != "count_y"], {"count_x": "Stones Across", "gap": "Gap Size", "bevel": "Roundness"}),
    "CONCRETE": (["count_x", "bump", "detail", "color1", "color2", "rough"], {"count_x": "Scale", "detail": "Pits & Cracks"}),
    "ROCK": (["count_x", "bump", "detail", "color1", "color2", "rough"], {"count_x": "Scale", "detail": "Specks"}),
    "METAL": (_PIECES + ["detail", "rivets"], {"count_x": "Plates Across", "count_y": "Plates Down", "gap": "Seam Size", "detail": "Scratches", "gap_color": "Seam Color"}),
    "FABRIC": (["count_x", "count_y", "bump", "detail", "color1", "color2", "variation", "rough"], {"count_x": "Threads Across", "count_y": "Threads Down", "detail": "Fibers", "color1": "Warp Color", "color2": "Weft Color"}),
    "ROOF": (["count_x", "count_y", "gap", "bevel", "wear", "bump", "detail", "color1", "color2", "gap_color", "variation", "rough", "rough_gap"],
             {"count_x": "Tiles per Row", "count_y": "Rows", "gap": "Side Gap", "bevel": "Round Bottom", "wear": "Edge Wear", "detail": "Barrel Curve", "gap_color": "Gap Color"}),
    "HERRINGBONE": (["count_x", "count_y", "gap", "bevel", "wear", "bump", "detail", "color1", "color2", "gap_color", "variation", "rough", "rough_gap"],
                    {"count_x": "Planks Across", "count_y": "Plank Length", "gap": "Gap Size", "detail": "Wood Grain", "color1": "Light Color", "color2": "Dark Color"}),
    "CORRUGATED": (["count_x", "count_y", "bump", "detail", "color1", "color2", "variation", "rough"],
                   {"count_x": "Ridges", "count_y": "Sheets Down", "bump": "Dents", "detail": "Rust", "color1": "Metal Color", "color2": "Rust Color"}),
    "RUSTY": (["count_x", "bump", "detail", "color1", "color2", "variation", "rough"],
              {"count_x": "Rust Scale", "bump": "Rust Roughness", "detail": "Rust Amount", "color1": "Paint Color", "color2": "Rust Color", "rough": "Paint Roughness"}),
    "BARK": (["count_x", "count_y", "bump", "detail", "color1", "color2", "variation", "rough"],
             {"count_x": "Ridges Across", "count_y": "Ridge Length", "bump": "Fibers", "detail": "Breaks", "color1": "Ridge Color", "color2": "Furrow Color"}),
    "GRAVEL": (["count_x", "gap", "bevel", "bump", "color1", "color2", "gap_color", "variation", "rough", "rough_gap"],
               {"count_x": "Pebbles Across", "gap": "Dirt Between", "bevel": "Roundness", "gap_color": "Dirt Color"}),
    "SAND": (["count_x", "bump", "detail", "color1", "color2", "variation", "rough"],
             {"count_x": "Ripples", "bump": "Ripple Height", "detail": "Grain", "color1": "Sand Color", "color2": "Trough Color"}),
    "MUD": (["count_x", "gap", "bevel", "wear", "bump", "detail", "color1", "color2", "gap_color", "variation", "rough", "rough_gap"],
            {"count_x": "Plates Across", "gap": "Crack Width", "bevel": "Edge Curl", "wear": "Crack Wobble", "detail": "Small Cracks", "gap_color": "Crack Color"}),
    "ASPHALT": (["count_x", "wear", "bump", "detail", "color1", "color2", "rough"],
                {"count_x": "Crack Scale", "wear": "Cracks", "detail": "Stones", "color1": "Asphalt Color", "color2": "Stone Color"}),
    "PLASTER": (["count_x", "wear", "bump", "detail", "color1", "color2", "variation", "rough"],
                {"count_x": "Scale", "wear": "Pits", "bump": "Trowel Marks", "detail": "Stucco Bumps"}),
    "TERRAZZO": (["count_x", "gap", "bump", "detail", "color1", "color2", "gap_color", "variation", "rough"],
                 {"count_x": "Chip Size (inverse)", "gap": "Chip Spacing", "detail": "Chip Amount", "color1": "Base Color", "color2": "Chip Color 1", "gap_color": "Chip Color 2"}),
    "LEATHER": (["count_x", "bump", "detail", "color1", "color2", "variation", "rough"],
                {"count_x": "Grain Size (inverse)", "bump": "Pebble Depth", "detail": "Creases"}),
}

# Good starting values loaded whenever the pattern is changed.
PRESETS = {
    "BRICKS": dict(count_x=4, count_y=12, offset=0.5, gap=0.12, bevel=0.15, wear=0.5, bump=0.4, detail=0.3,
                   color1=(0.58, 0.27, 0.19), color2=(0.45, 0.22, 0.17), gap_color=(0.66, 0.63, 0.58),
                   variation=0.25, rough=0.8, rough_gap=0.95),
    "TILES": dict(count_x=6, count_y=6, gap=0.06, bevel=0.1, wear=0.03, bump=0.08, detail=0.3,
                  color1=(0.88, 0.88, 0.85), color2=(0.80, 0.82, 0.82), gap_color=(0.55, 0.53, 0.50),
                  variation=0.05, rough=0.15, rough_gap=0.9),
    "HEX": dict(count_x=7, count_y=4, gap=0.06, bevel=0.15, wear=0.03, bump=0.08, detail=0.3,
                color1=(0.20, 0.36, 0.46), color2=(0.85, 0.85, 0.80), gap_color=(0.60, 0.58, 0.55),
                variation=0.08, rough=0.2, rough_gap=0.9),
    "PLANKS": dict(count_x=2, count_y=8, gap=0.04, bevel=0.08, wear=0.1, bump=0.2, detail=0.7,
                   color1=(0.62, 0.43, 0.25), color2=(0.36, 0.21, 0.11), gap_color=(0.10, 0.07, 0.05),
                   variation=0.2, rough=0.55, rough_gap=0.9),
    "STONES": dict(count_x=6, gap=0.08, bevel=0.6, wear=0.6, bump=0.6, detail=0.3,
                   color1=(0.55, 0.53, 0.50), color2=(0.40, 0.38, 0.36), gap_color=(0.30, 0.27, 0.23),
                   variation=0.3, rough=0.85, rough_gap=0.95),
    "CONCRETE": dict(count_x=4, bump=0.5, detail=0.5, color1=(0.62, 0.62, 0.60), color2=(0.50, 0.50, 0.48), rough=0.9),
    "ROCK": dict(count_x=4, bump=1.0, detail=0.5, color1=(0.33, 0.30, 0.28), color2=(0.62, 0.58, 0.53), rough=0.85),
    "METAL": dict(count_x=2, count_y=2, gap=0.02, bevel=0.03, wear=0.05, bump=0.1, detail=0.5,
                  color1=(0.65, 0.66, 0.68), color2=(0.56, 0.57, 0.59), gap_color=(0.15, 0.15, 0.15),
                  variation=0.05, rough=0.35, rough_gap=0.7, rivets=True),
    "FABRIC": dict(count_x=48, count_y=48, bump=0.3, detail=0.5, variation=0.1,
                   color1=(0.20, 0.30, 0.55), color2=(0.15, 0.22, 0.42), rough=0.9),
    "ROOF": dict(count_x=6, count_y=10, gap=0.05, bevel=0.6, wear=0.4, bump=0.3, detail=0.5,
                 color1=(0.62, 0.30, 0.20), color2=(0.48, 0.24, 0.17), gap_color=(0.12, 0.08, 0.06),
                 variation=0.2, rough=0.7, rough_gap=0.9),
    "HERRINGBONE": dict(count_x=16, count_y=4, gap=0.06, bevel=0.1, wear=0.1, bump=0.2, detail=0.6,
                        color1=(0.66, 0.47, 0.28), color2=(0.42, 0.26, 0.14), gap_color=(0.12, 0.08, 0.05),
                        variation=0.25, rough=0.45, rough_gap=0.9),
    "CORRUGATED": dict(count_x=16, count_y=2, bump=0.4, detail=0.4, variation=0.1,
                       color1=(0.62, 0.64, 0.66), color2=(0.45, 0.22, 0.10), rough=0.35),
    "RUSTY": dict(count_x=4, bump=0.6, detail=0.45, variation=0.15,
                  color1=(0.22, 0.38, 0.30), color2=(0.42, 0.20, 0.09), rough=0.45),
    "BARK": dict(count_x=12, count_y=3, bump=0.7, detail=0.6, variation=0.2,
                 color1=(0.42, 0.34, 0.27), color2=(0.17, 0.12, 0.09), rough=0.9),
    "GRAVEL": dict(count_x=18, gap=0.15, bevel=0.6, bump=0.3, variation=0.35,
                   color1=(0.62, 0.58, 0.52), color2=(0.38, 0.36, 0.34), gap_color=(0.28, 0.22, 0.16),
                   rough=0.75, rough_gap=0.95),
    "SAND": dict(count_x=10, bump=0.6, detail=0.6, variation=0.15,
                 color1=(0.83, 0.70, 0.50), color2=(0.70, 0.56, 0.38), rough=0.95),
    "MUD": dict(count_x=7, gap=0.12, bevel=0.4, wear=0.6, bump=0.4, detail=0.4, variation=0.15,
                color1=(0.58, 0.45, 0.32), color2=(0.50, 0.38, 0.27), gap_color=(0.18, 0.13, 0.09),
                rough=0.9, rough_gap=0.95),
    "ASPHALT": dict(count_x=3, wear=0.4, bump=0.4, detail=0.6,
                    color1=(0.16, 0.16, 0.17), color2=(0.55, 0.53, 0.50), rough=0.9),
    "PLASTER": dict(count_x=4, wear=0.3, bump=0.9, detail=0.6, variation=0.2,
                    color1=(0.86, 0.83, 0.77), color2=(0.78, 0.74, 0.67), rough=0.85),
    "TERRAZZO": dict(count_x=14, gap=0.5, bump=0.2, detail=0.4, variation=0.2,
                     color1=(0.88, 0.86, 0.82), color2=(0.45, 0.43, 0.42), gap_color=(0.72, 0.42, 0.33),
                     rough=0.2),
    "LEATHER": dict(count_x=90, bump=0.5, detail=0.4, variation=0.15,
                    color1=(0.36, 0.20, 0.11), color2=(0.28, 0.15, 0.08), rough=0.55),
}


def _apply_preset(self, context):
    for key, value in PRESETS.get(self.tex_pattern, {}).items():
        setattr(self, "tex_" + key, value)


# ---------------------------------------------------------------------------
# Material builder
# ---------------------------------------------------------------------------

def _socket(sockets, identifier, fallback_index):
    for s in sockets:
        if s.identifier == identifier:
            return s
    return sockets[fallback_index]


def build_material(name, maps, props):
    """maps: dict with keys albedo, height, normal, roughness, ao, [metallic] (Images)."""
    mat = bpy.data.materials.new(name)
    if mat.node_tree is None or not getattr(mat, "use_nodes", True):
        try:
            mat.use_nodes = True
        except AttributeError:
            pass
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (900, 0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (550, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    use_box = getattr(props, "projection", "UV") == "BOX"
    coord = nodes.new("ShaderNodeTexCoord")
    coord.location = (-1100, 0)
    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-900, 0)
    if use_box:
        # Object coordinates: at Tiling 1 the texture repeats once every 2 m
        # (exactly once per face of the default cube).
        s = props.tiling / 2.0
        mapping.inputs["Scale"].default_value = (s, s, s)
        mapping.inputs["Location"].default_value = (0.5, 0.5, 0.5)
        links.new(coord.outputs["Object"], mapping.inputs["Vector"])
    else:
        mapping.inputs["Scale"].default_value = (props.tiling, props.tiling, 1.0)
        links.new(coord.outputs["UV"], mapping.inputs["Vector"])

    def tex(img, label, y):
        n = nodes.new("ShaderNodeTexImage")
        n.image = img
        n.label = label
        n.location = (-600, y)
        if use_box:
            n.projection = "BOX"
            n.projection_blend = props.box_blend
        links.new(mapping.outputs["Vector"], n.inputs["Vector"])
        return n

    albedo = tex(maps["albedo"], "Base Color", 400)
    ao = tex(maps["ao"], "Ambient Occlusion", 120)
    rough = tex(maps["roughness"], "Roughness", -160)
    normal = tex(maps["normal"], "Normal", -440)
    height = tex(maps["height"], "Height / Bump", -720)

    # Base color multiplied by AO
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.label = "AO Multiply"
    mix.location = (-50, 300)
    mix.inputs[0].default_value = props.ao_mix
    links.new(albedo.outputs["Color"], _socket(mix.inputs, "A_Color", 6))
    links.new(ao.outputs["Color"], _socket(mix.inputs, "B_Color", 7))
    links.new(_socket(mix.outputs, "Result_Color", 2), bsdf.inputs["Base Color"])

    links.new(rough.outputs["Color"], bsdf.inputs["Roughness"])

    if maps.get("metallic") is not None:
        metal = tex(maps["metallic"], "Metallic", 680)
        links.new(metal.outputs["Color"], bsdf.inputs["Metallic"])

    if use_box:
        # Normal maps rely on UV directions, so box mode builds the bumps from
        # the height map instead (works with any projection).
        normal.label = "Normal (used in UV mode)"
        bump = nodes.new("ShaderNodeBump")
        bump.location = (-50, -600)
        bump.inputs["Strength"].default_value = 1.0
        bump.inputs["Distance"].default_value = 0.008 * props.normal_strength
        links.new(height.outputs["Color"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    else:
        nmap = nodes.new("ShaderNodeNormalMap")
        nmap.location = (-50, -440)
        nmap.inputs["Strength"].default_value = 1.0
        links.new(normal.outputs["Color"], nmap.inputs["Color"])
        links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

    disp = nodes.new("ShaderNodeDisplacement")
    disp.location = (550, -600)
    disp.inputs["Midlevel"].default_value = 0.5
    disp.inputs["Scale"].default_value = props.displacement_scale
    links.new(height.outputs["Color"], disp.inputs["Height"])
    if props.use_displacement:
        links.new(disp.outputs["Displacement"], out.inputs["Displacement"])
        if hasattr(mat, "displacement_method"):
            mat.displacement_method = "BOTH"

    return mat


def assign_to_active(context, mat):
    obj = context.active_object
    if obj is None or not hasattr(obj.data, "materials"):
        return False
    if obj.data.materials and obj.active_material_index < len(obj.data.materials):
        obj.data.materials[obj.active_material_index] = mat
    else:
        obj.data.materials.append(mat)
    return True


def finish(op, context, base, maps, props, replace=False):
    old = bpy.data.materials.get(base) if replace else None
    mat = build_material(base, maps, props)
    if replace:
        replace_datablock(old, mat, base)
    msg = ("Replaced material '%s'" if old is not None else "Created material '%s'") % mat.name
    if props.assign_to_active:
        if assign_to_active(context, mat):
            obj = context.active_object
            msg += " on '%s'" % obj.name
            if props.projection == "UV" and obj.type == "MESH" and not obj.data.uv_layers:
                op.report({"WARNING"}, "Object has no UV map; unwrap it (U) to see the texture")
        else:
            msg += " (no suitable active object to assign it to)"
    op.report({"INFO"}, msg)


# ---------------------------------------------------------------------------
# Image download
# ---------------------------------------------------------------------------

_EXT_BY_TYPE = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tif",
    "image/x-exr": ".exr",
}


def download_image(url, out_dir=""):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("URL must start with http:// or https://")
    req = urllib.request.Request(url, headers={"User-Agent": "Blender-PBR-Map-Generator/1.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        ctype = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
        data = resp.read()
    if ctype and not ctype.startswith("image/"):
        raise ValueError("That link is a web page (%s), not an image. "
                         "Right-click the image and use 'Copy image address'." % ctype)
    ext = os.path.splitext(parsed.path)[1].lower()
    if ext not in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".exr", ".tga"):
        ext = _EXT_BY_TYPE.get(ctype, ".png")
    stem = bpy.path.clean_name(os.path.splitext(os.path.basename(parsed.path))[0] or "download")
    folder = bpy.path.abspath(out_dir) if out_dir else os.path.join(tempfile.gettempdir(), "pbr_map_downloads")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, stem + ext)
    with open(path, "wb") as f:
        f.write(data)
    img = bpy.data.images.load(path, check_existing=False)
    img.pack()  # keep it inside the .blend
    return img


# ---------------------------------------------------------------------------
# Live preview: scheduling
# ---------------------------------------------------------------------------

preview_collections = {}
_preview_state = {"counter": 0, "last_path": ""}


def _run_preview_timer():
    try:
        wm = bpy.context.window_manager
        if wm.windows:
            with bpy.context.temp_override(window=wm.windows[0]):
                bpy.ops.pbrgen.preview()
        else:
            bpy.ops.pbrgen.preview()
    except Exception as e:  # noqa: BLE001
        print("PBR Map Generator: preview failed:", e)
    return None


def _queue_preview(context, force=False):
    """Re-render the preview shortly after a setting changes (debounced)."""
    if bpy.app.background:
        return
    scene = context.scene if context else bpy.context.scene
    if not force and not scene.pbrgen.preview_live:
        return
    if bpy.app.timers.is_registered(_run_preview_timer):
        bpy.app.timers.unregister(_run_preview_timer)
    bpy.app.timers.register(_run_preview_timer, first_interval=0.4)


def schedule_preview(self, context):
    _queue_preview(context)


def _on_shape_change(self, context):
    pcoll = preview_collections.get("main")
    if self.preview_live or (pcoll is not None and "preview" in pcoll):
        _queue_preview(context, force=True)


def _redraw_viewports(self=None, context=None):
    wm = bpy.context.window_manager
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


PREVIEW_IMAGE_NAME = ".PBRGen Preview"  # leading dot hides it from image lists
_draw_state = {"handle": None, "visible": {}, "debug": {}}

def sidebar_icon_scale(region_width, ui):
    """Icon scale that makes the sidebar preview fill the sidebar's width."""
    return max(8.0, (region_width / ui - 56) / 20.0)


def sidebar_preview_bottom(region_width, ui):
    """Roughly where the sidebar preview image ends, measured from the top of the
    sidebar. Once scrolled past this, the image is off screen and the
    viewport-corner copy is shown instead."""
    return (40 + 20 * sidebar_icon_scale(region_width, ui)) * ui


def sidebar_preview_visible(area):
    """True when this 3D view's sidebar is open on our tab and scrolled
    near the top, so the sidebar's own preview image is on screen."""
    space = area.spaces.active
    if not getattr(space, "show_region_ui", False):
        return False
    for r in area.regions:
        if r.type != "UI" or r.width <= 1:
            continue
        if getattr(r, "active_panel_category", "PBR Maps") != "PBR Maps":
            return False
        try:
            top = r.view2d.region_to_view(0, r.height)[1]
        except Exception:  # noqa: BLE001
            return False
        scrolled = -top
        _draw_state["debug"]["scrolled"] = scrolled
        ui = bpy.context.preferences.system.ui_scale
        return scrolled < sidebar_preview_bottom(r.width, ui) - 30 * ui
    return False


def sync_viewport_overlay(context):
    """Called while the sidebar draws (which includes scrolling). If the sidebar
    preview just scrolled in or out of view, redraw the viewport so the corner
    preview hides or shows to match."""
    area = context.area
    if area is None or area.type != "VIEW_3D":
        return
    visible = sidebar_preview_visible(area)
    key = area.as_pointer()
    if _draw_state["visible"].get(key) == visible:
        return
    _draw_state["visible"][key] = visible

    def _tag():
        try:
            for r in area.regions:
                if r.type == "WINDOW":
                    r.tag_redraw()
        except ReferenceError:
            pass
        return None
    bpy.app.timers.register(_tag, first_interval=0.0)


def _draw_viewport_preview():
    """Draws the latest preview in the bottom-right corner of every 3D viewport."""
    ctx = bpy.context
    scene = ctx.scene
    if scene is None or not hasattr(scene, "pbrgen") or not scene.pbrgen.preview_in_viewport:
        return
    img = bpy.data.images.get(PREVIEW_IMAGE_NAME)
    region, area = ctx.region, ctx.area
    if img is None or region is None or area is None:
        return
    if sidebar_preview_visible(area):
        _draw_state["debug"]["overlay_drawn"] = False
        return  # the sidebar already shows it; don't show two previews
    _draw_state["debug"]["overlay_drawn"] = True
    try:
        texture = gpu.texture.from_image(img)
    except Exception:  # noqa: BLE001 - e.g. temp file was cleaned up
        return

    prefs = ctx.preferences.system
    ui = prefs.ui_scale
    size = scene.pbrgen.preview_viewport_size * ui
    margin = 14 * ui
    right = 0
    if prefs.use_region_overlap and area.spaces.active.show_region_ui:
        for r in area.regions:
            if r.type == "UI" and r.width > 1:
                right = r.width
    x1 = region.width - right - margin
    x0 = max(margin, x1 - size)
    y0 = margin
    y1 = y0 + (x1 - x0)

    gpu.state.blend_set("ALPHA")
    frame = gpu.shader.from_builtin("UNIFORM_COLOR")
    b = 2 * ui
    quad = ((0, 1, 2), (2, 3, 0))
    frame.bind()
    frame.uniform_float("color", (0.0, 0.0, 0.0, 0.6))
    batch_for_shader(frame, "TRIS", {"pos": ((x0 - b, y0 - b), (x1 + b, y0 - b), (x1 + b, y1 + b), (x0 - b, y1 + b))},
                     indices=quad).draw(frame)

    shader = gpu.shader.from_builtin("IMAGE")
    batch = batch_for_shader(shader, "TRIS",
                             {"pos": ((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
                              "texCoord": ((0, 0), (1, 0), (1, 1), (0, 1))},
                             indices=quad)
    shader.bind()
    shader.uniform_sampler("image", texture)
    batch.draw(shader)
    gpu.state.blend_set("NONE")


def _on_pattern_change(self, context):
    _apply_preset(self, context)
    schedule_preview(self, context)


# ---------------------------------------------------------------------------
# Properties (every description below is the hover tooltip in Blender)
# ---------------------------------------------------------------------------

def _col(name, default, desc):
    return FloatVectorProperty(name=name, subtype="COLOR_GAMMA", size=3, default=default,
                               min=0.0, max=1.0, description=desc, update=schedule_preview)


class PBRGEN_Props(bpy.types.PropertyGroup):
    # --- preview
    preview_shape: EnumProperty(
        name="Shape",
        description="Which object the preview shows the material on",
        items=[
            ("CUBE", "Cube", "Preview on a cube", "MESH_CUBE", 0),
            ("SPHERE", "Sphere", "Preview on a UV sphere", "MESH_UVSPHERE", 1),
            ("CYLINDER", "Cylinder", "Preview on a cylinder", "MESH_CYLINDER", 2),
            ("CONE", "Cone", "Preview on a cone", "MESH_CONE", 3),
            ("TORUS", "Torus", "Preview on a torus (donut)", "MESH_TORUS", 4),
            ("MONKEY", "Suzanne", "Preview on Suzanne, Blender's monkey head", "MESH_MONKEY", 5),
            ("PLANE", "Plane", "Preview on a flat plane, good for floors and walls", "MESH_PLANE", 6),
        ],
        default="SPHERE",
        update=_on_shape_change)
    preview_source: EnumProperty(
        name="Preview",
        description="What the preview shows",
        items=[
            ("TEXTURE", "Texture Generator", "Preview the pattern from the Texture Generator panel"),
            ("IMAGE", "Image Maps", "Preview the maps made from the image in the Maps from Image panel"),
        ],
        default="TEXTURE",
        update=schedule_preview)
    preview_view: EnumProperty(
        name="View",
        description="How the preview shows the texture",
        items=[
            ("OBJECT", "Object", "The whole shape, to judge the overall look"),
            ("CLOSEUP", "Close-up", "Zoomed in on the surface, to judge bumps, roughness and fine detail"),
            ("MAPS", "Maps", "The Color, Normal, Roughness and Height maps side by side, "
                             "so you can see exactly which map a setting changes"),
        ],
        default="OBJECT",
        update=schedule_preview)
    on_existing: EnumProperty(
        name="If It Already Exists",
        description="What to do when you generate a texture whose material already exists "
                    "(for example generating Wood Planks again with different settings)",
        items=[
            ("ASK", "Ask Me", "Show a choice each time: replace the existing texture or create a new one"),
            ("REPLACE", "Replace", "Overwrite the existing material and maps. Every object using it updates"),
            ("NEW", "Create New", "Always make a separate new material (e.g. Bricks_PBR.001) and keep the old one"),
        ],
        default="ASK")
    preview_live: BoolProperty(
        name="Auto Update", default=True,
        description="Re-render the preview automatically whenever you change a setting. "
                    "Turn off if your computer feels slow while adjusting sliders")
    preview_in_viewport: BoolProperty(
        name="Show in Viewport", default=True, update=_redraw_viewports,
        description="Also show the preview in the bottom-right corner of the 3D viewport, "
                    "so it stays visible while you scroll through the settings")
    preview_viewport_size: IntProperty(
        name="Viewport Preview Size", default=220, min=96, max=640, subtype="PIXEL", update=_redraw_viewports,
        description="How big the preview is in the corner of the 3D viewport")

    # --- shared output settings
    out_dir: StringProperty(
        name="Save Folder", subtype="DIR_PATH", default="",
        description="Folder to save the generated maps into as PNG files. "
                    "Leave empty to store them inside the .blend file instead")
    normal_strength: FloatProperty(
        name="Normal Strength", default=4.0, min=0.0, soft_max=30.0, update=schedule_preview,
        description="How deep the bumps look in the normal map. Higher = stronger, more dramatic surface detail")
    directx_normal: BoolProperty(
        name="DirectX Normals (flip green)", default=False, update=schedule_preview,
        description="Flip the normal map's green channel for DirectX engines such as Unreal. "
                    "Leave off for Blender, Unity and Godot (they use OpenGL normals)")
    ao_strength: FloatProperty(
        name="AO Strength", default=4.0, min=0.0, soft_max=20.0, update=schedule_preview,
        description="How dark the ambient occlusion makes cracks, gaps and crevices")
    ao_radius: IntProperty(
        name="AO Radius", default=12, min=1, soft_max=64, subtype="PIXEL", update=schedule_preview,
        description="How far the ambient occlusion shadow spreads out from crevices, in pixels")
    ao_mix: FloatProperty(
        name="AO in Material", default=0.6, min=0.0, max=1.0, subtype="FACTOR", update=schedule_preview,
        description="How much the AO map darkens the base color in the material. 0 = not used, 1 = full effect")
    tiling: FloatProperty(
        name="Tiling", default=1.0, min=0.01, soft_max=20.0, update=schedule_preview,
        description="How many times the texture repeats across the object's UVs. "
                    "Raise it to make bricks, planks etc. smaller on the object")
    projection: EnumProperty(
        name="Projection",
        description="How the texture is wrapped onto the object",
        items=[
            ("BOX", "Box", "Wrap the texture onto each side of the object by its position. No UV unwrapping "
                           "needed, and pieces stay the same size on every face. Bumps come from the height map"),
            ("UV", "UV", "Use the object's UV map. Needed for exporting to game engines. "
                         "The object must be unwrapped well (e.g. U > Cube Projection or Smart UV Project)"),
        ],
        default="BOX",
        update=schedule_preview)
    box_blend: FloatProperty(
        name="Edge Blend", default=0.15, min=0.0, max=1.0, subtype="FACTOR", update=schedule_preview,
        description="Box projection only: how softly the texture blends where two sides meet. "
                    "0 = sharp seam, higher = smoother blend on rounded objects")
    use_displacement: BoolProperty(
        name="Use Displacement", default=False, update=schedule_preview,
        description="Plug the height map into the material's displacement so the surface really moves. "
                    "Needs a finely subdivided mesh to look right")
    displacement_scale: FloatProperty(
        name="Displacement Scale", default=0.05, min=0.0, soft_max=1.0, update=schedule_preview,
        description="How far the displacement pushes the surface out, in meters")
    assign_to_active: BoolProperty(
        name="Assign to Active Object", default=True,
        description="Put the new material on the selected object after generating")

    # --- image source
    source_mode: EnumProperty(
        name="Source",
        description="Where the source image comes from",
        items=[
            ("FILE", "File", "Load an image file from your computer"),
            ("URL", "URL", "Download an image from the internet"),
            ("EXISTING", "Loaded", "Use an image that is already loaded in Blender"),
        ],
        default="FILE")
    filepath: StringProperty(
        name="Image File", subtype="FILE_PATH", default="", update=schedule_preview,
        description="The photo or texture image to turn into PBR maps")
    url: StringProperty(
        name="Image URL", default="",
        description="Direct link to an image online. In your browser, right-click the image and "
                    "choose 'Copy image address', then paste it here")
    source_image: PointerProperty(
        name="Image", type=bpy.types.Image, update=schedule_preview,
        description="An image already loaded in Blender to turn into PBR maps")
    max_size: IntProperty(
        name="Max Resolution", default=2048, min=64, max=8192, subtype="PIXEL",
        description="Images bigger than this are scaled down first. Lower = faster, higher = more detail")

    height_blur: IntProperty(
        name="Height Smoothing", default=2, min=0, soft_max=20, subtype="PIXEL", update=schedule_preview,
        description="Blurs the height map to remove grainy noise. Higher = softer, smoother bumps")
    height_contrast: FloatProperty(
        name="Height Contrast", default=1.0, min=0.0, soft_max=4.0, update=schedule_preview,
        description="Exaggerates the difference between high and low areas of the surface")
    invert_height: BoolProperty(
        name="Invert Height", default=False, update=schedule_preview,
        description="Swap raised and sunken areas. Use it if gaps or cracks are coming out raised")
    detail_balance: FloatProperty(
        name="Large Shape Removal", default=0.5, min=0.0, max=1.0, subtype="FACTOR", update=schedule_preview,
        description="Removes big light-to-dark gradients (like uneven lighting) from the height, "
                    "keeping only the small surface detail")
    rough_min: FloatProperty(
        name="Roughness Min", default=0.35, min=0.0, max=1.0, update=schedule_preview,
        description="Roughness of the shiniest parts. 0 = mirror-like, 1 = completely matte")
    rough_max: FloatProperty(
        name="Roughness Max", default=0.9, min=0.0, max=1.0, update=schedule_preview,
        description="Roughness of the most matte parts. 0 = mirror-like, 1 = completely matte")
    invert_roughness: BoolProperty(
        name="Invert Roughness", default=False, update=schedule_preview,
        description="Swap shiny and matte areas. By default dark parts of the image are treated as rougher")
    delight: BoolProperty(
        name="Even Out Lighting", default=True, update=schedule_preview,
        description="Removes big shadows and bright spots that were baked into the photo, "
                    "so the color looks evenly lit")

    # --- procedural textures
    tex_pattern: EnumProperty(
        name="Pattern",
        items=_pattern_enum_items(),
        default="BRICKS",
        update=_on_pattern_change,
        description="What kind of texture to generate. Changing it loads good starting settings")
    tex_res: EnumProperty(
        name="Resolution",
        items=[("512", "512", "512 x 512 pixels: fast, low detail"),
               ("1024", "1024", "1024 x 1024 pixels: good default"),
               ("2048", "2048", "2048 x 2048 pixels: high detail"),
               ("4096", "4096", "4096 x 4096 pixels: very high detail, slow and uses lots of memory")],
        default="1024",
        description="Size of the generated texture images (the preview always uses a small size)")
    tex_count_x: IntProperty(
        name="Count X", default=4, min=1, max=512, update=schedule_preview,
        description="How many pieces fit across the texture (bricks, tiles, planks, stones, plates or threads). "
                    "For Concrete and Rock this is the noise scale: higher = smaller features")
    tex_count_y: IntProperty(
        name="Count Y", default=12, min=1, max=512, update=schedule_preview,
        description="How many rows fit down the texture. For bricks, use an even number so it repeats seamlessly")
    tex_offset: FloatProperty(
        name="Row Offset", default=0.5, min=0.0, max=1.0, subtype="FACTOR", update=schedule_preview,
        description="How far every other row is shifted sideways. 0.5 = classic brick wall, 0 = stacked in a grid")
    tex_gap: FloatProperty(
        name="Gap Size", default=0.12, min=0.0, max=0.9, subtype="FACTOR", update=schedule_preview,
        description="Width of the mortar, grout or seam between pieces")
    tex_bevel: FloatProperty(
        name="Bevel", default=0.15, min=0.0, max=1.0, subtype="FACTOR", update=schedule_preview,
        description="How rounded the edges of each piece are. Higher = softer, more rounded edges")
    tex_wear: FloatProperty(
        name="Edge Wear", default=0.5, min=0.0, max=2.0, update=schedule_preview,
        description="How chipped and uneven the edges are. 0 = perfectly clean straight edges")
    tex_bump: FloatProperty(
        name="Surface Bumpiness", default=0.4, min=0.0, max=2.0, update=schedule_preview,
        description="How lumpy and uneven the surface of each piece is")
    tex_detail: FloatProperty(
        name="Detail", default=0.3, min=0.0, max=1.0, subtype="FACTOR", update=schedule_preview,
        description="Extra pattern detail: wood grain for planks, pits and cracks for concrete, "
                    "specks for rock, scratches for metal, fibers for fabric")
    tex_rivets: BoolProperty(
        name="Rivets", default=True, update=schedule_preview,
        description="Add a rivet near each corner of every metal plate")
    tex_color1: _col("Color", (0.58, 0.27, 0.19),
                     "Main color of the pieces (warp threads for fabric, low areas for rock)")
    tex_color2: _col("Color 2", (0.45, 0.22, 0.17),
                     "Second color. Each piece randomly picks or blends between Color and Color 2")
    tex_gap_color: _col("Gap Color", (0.66, 0.63, 0.58),
                        "Color of the mortar, grout or seams between pieces")
    tex_variation: FloatProperty(
        name="Color Variation", default=0.25, min=0.0, max=1.0, subtype="FACTOR", update=schedule_preview,
        description="How much each piece's brightness randomly differs from its neighbors")
    tex_rough: FloatProperty(
        name="Roughness", default=0.8, min=0.0, max=1.0, update=schedule_preview,
        description="How matte the pieces are. 0 = glossy like glazed tile, 1 = completely matte like raw stone")
    tex_rough_gap: FloatProperty(
        name="Gap Roughness", default=0.95, min=0.0, max=1.0, update=schedule_preview,
        description="How matte the mortar, grout or seams are")
    tex_seed: IntProperty(
        name="Seed", default=1, min=0, update=schedule_preview,
        description="Change this to get a different random version of the same texture")


# ---------------------------------------------------------------------------
# Map generation (arrays only, no Blender data)
# ---------------------------------------------------------------------------

def read_rgb(img, max_size):
    w, h = img.size
    if w == 0 or h == 0:
        raise ValueError("Image has no pixel data (is the file missing?)")
    scale = min(1.0, max_size / float(max(w, h)))
    if scale < 1.0:
        tmp = img.copy()
        try:
            tmp.scale(max(1, int(w * scale)), max(1, int(h * scale)))
            px = image_to_array(tmp)
        finally:
            bpy.data.images.remove(tmp)
    else:
        px = image_to_array(img)
    return np.clip(px[..., :3], 0.0, 1.0)


def image_arrays(rgb, p):
    h, w = rgb.shape[:2]
    big = max(4, int(max(h, w) / 16))
    lum = luminance(rgb)

    albedo = rgb
    if p.delight:
        low = gauss_blur(lum, big)
        gain = np.clip(float(lum.mean()) / np.maximum(low, 1e-3), 0.5, 2.0)
        albedo = np.clip(rgb * gain[..., None], 0.0, 1.0)

    height = lum.copy()
    if p.detail_balance > 0:
        height = height - gauss_blur(height, big) * p.detail_balance
    if p.height_blur:
        height = gauss_blur(height, p.height_blur)
    height = normalize01(height)
    if p.invert_height:
        height = 1.0 - height
    height = apply_contrast(height, p.height_contrast)

    normal = height_to_normal(height, p.normal_strength * max(h, w) / 512.0, p.directx_normal)

    detail = np.abs(lum - gauss_blur(lum, 3))
    rough = normalize01((1.0 - normalize01(lum)) * 0.7 + normalize01(detail) * 0.3)
    if p.invert_roughness:
        rough = 1.0 - rough
    rough = p.rough_min + rough * (p.rough_max - p.rough_min)

    ao = height_to_ao(height, max(1, int(p.ao_radius * max(h, w) / 1024.0)), p.ao_strength)
    return {"albedo": albedo, "height": height, "normal": normal, "roughness": rough, "ao": ao}


def procedural_arrays(p, size):
    func = PATTERNS[p.tex_pattern][1]
    data = func(p, size, np.random.default_rng(p.tex_seed))
    height = np.clip(data["height"], 0, 1)
    normal = height_to_normal(height, p.normal_strength * size / 1024.0 * 2.0, p.directx_normal)
    ao = height_to_ao(height, max(2, int(p.ao_radius * size / 1024.0)), p.ao_strength * 0.5)
    albedo = np.clip(data["albedo"], 0, 1) * (0.75 + 0.25 * ao[..., None])
    arrays = {"albedo": albedo, "height": height, "normal": normal,
              "roughness": np.clip(data["rough"], 0, 1), "ao": ao}
    if "metal" in data:
        arrays["metallic"] = data["metal"]
    return arrays


def get_source_image(op, p):
    """Resolve the image chosen in the 'Maps from Image' panel (may download)."""
    if p.source_mode == "FILE":
        path = bpy.path.abspath(p.filepath)
        if not p.filepath or not os.path.isfile(path):
            op.report({"ERROR"}, "Choose an image file first")
            return None
        return bpy.data.images.load(path, check_existing=True)
    if p.source_mode == "URL":
        if not p.url.strip():
            op.report({"ERROR"}, "Paste an image URL first")
            return None
        if not getattr(bpy.app, "online_access", True):
            op.report({"ERROR"}, "Online access is disabled. Enable it in Preferences > System > Network")
            return None
        img = download_image(p.url.strip(), p.out_dir)
        p.source_image = img
        p.source_mode = "EXISTING"  # don't download again on every preview
        return img
    if p.source_image is None:
        op.report({"ERROR"}, "Pick a loaded image first")
    return p.source_image


# ---------------------------------------------------------------------------
# Preview rendering
# ---------------------------------------------------------------------------

def _torus(bm, uv, major=1.0, minor=0.42, seg=64, ring=32):
    verts = [[bm.verts.new((
        (major + minor * math.cos(2 * math.pi * j / ring)) * math.cos(2 * math.pi * i / seg),
        (major + minor * math.cos(2 * math.pi * j / ring)) * math.sin(2 * math.pi * i / seg),
        minor * math.sin(2 * math.pi * j / ring)))
        for j in range(ring)] for i in range(seg)]
    for i in range(seg):
        for j in range(ring):
            i2, j2 = (i + 1) % seg, (j + 1) % ring
            f = bm.faces.new((verts[i][j], verts[i2][j], verts[i2][j2], verts[i][j2]))
            coords = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))
            for loop, (a, b) in zip(f.loops, coords):
                loop[uv].uv = (a / seg * 3.0, b / ring)


def make_shape_mesh(shape):
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    if shape == "CUBE":
        # Same UV layout as Blender's default cube, so UV mode previews honestly.
        bmesh.ops.create_cube(bm, size=2.0, calc_uvs=True)
    elif shape == "SPHERE":
        bmesh.ops.create_uvsphere(bm, u_segments=64, v_segments=32, radius=1.25, calc_uvs=True)
    elif shape == "CYLINDER":
        bmesh.ops.create_cone(bm, cap_ends=True, segments=64, radius1=1.0, radius2=1.0,
                              depth=2.0, calc_uvs=True)
    elif shape == "CONE":
        bmesh.ops.create_cone(bm, cap_ends=True, segments=64, radius1=1.2, radius2=0.0,
                              depth=2.2, calc_uvs=True)
    elif shape == "TORUS":
        _torus(bm, uv)
    elif shape == "MONKEY":
        bmesh.ops.create_monkey(bm, calc_uvs=True)
        bmesh.ops.scale(bm, vec=(1.3, 1.3, 1.3), verts=bm.verts)
    else:  # PLANE
        bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=1.6, calc_uvs=True)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    smooth = shape not in ("CUBE", "PLANE")
    for f in bm.faces:
        f.smooth = smooth
    mesh = bpy.data.meshes.new("PBRGenPreview_Mesh")
    bm.to_mesh(mesh)
    bm.free()
    if smooth and hasattr(mesh, "set_sharp_from_angle"):
        mesh.set_sharp_from_angle(angle=math.radians(40))
    return mesh


def _set_engine(scene):
    for engine in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "CYCLES"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = 24
    elif hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = 24


def _aim(obj, target=(0.0, 0.0, 0.0)):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _frame_object(scene, cam, mesh, fill=0.8):
    """Zoom and shift the camera so the shape's on-screen outline is centred
    and fills the same fraction of the frame, whatever the shape."""
    cam.matrix_world = Matrix.Translation(cam.location) @ cam.rotation_euler.to_matrix().to_4x4()
    pts = [world_to_camera_view(scene, cam, v.co) for v in mesh.vertices]
    if not pts:
        return
    xs = [pt.x for pt in pts]
    ys = [pt.y for pt in pts]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    if w <= 0 or h <= 0:
        return
    k = fill / max(w, h)
    cam.data.lens *= k
    cam.data.shift_x = k * ((max(xs) + min(xs)) / 2 - 0.5)
    cam.data.shift_y = k * ((max(ys) + min(ys)) / 2 - 0.5)


def _next_preview_path():
    _preview_state["counter"] += 1
    return os.path.join(tempfile.gettempdir(), "pbrgen_preview_%d.png" % _preview_state["counter"])


def _resize(a, n):
    """Nearest-neighbour resize of a (H, W[, C]) array to n x n."""
    ys = np.linspace(0, a.shape[0] - 1, n).astype(np.int64)
    xs = np.linspace(0, a.shape[1] - 1, n).astype(np.int64)
    return a[np.ix_(ys, xs)] if a.ndim == 2 else a[ys][:, xs]


def maps_grid_png(arrays, size=512):
    """2 x 2 grid: Color | Normal on top, Roughness | Height below."""
    half = size // 2

    def tile(key):
        a = _resize(np.clip(arrays[key], 0, 1).astype(np.float32), half)
        return np.dstack([a, a, a]) if a.ndim == 2 else a[..., :3]

    top = np.concatenate([tile("albedo"), tile("normal")], axis=1)
    bottom = np.concatenate([tile("roughness"), tile("height")], axis=1)
    grid = np.concatenate([bottom, top], axis=0)  # row 0 is the bottom in Blender
    grid[half - 2:half + 2, :] = 0.12
    grid[:, half - 2:half + 2] = 0.12
    rgba = np.dstack([grid, np.ones((size, size), np.float32)])
    img = bpy.data.images.new("PBRGenPreview_Grid", size, size, alpha=True)
    try:
        img.pixels.foreach_set(rgba.ravel())
        path = _next_preview_path()
        img.filepath_raw = path
        img.file_format = "PNG"
        img.save()
    finally:
        bpy.data.images.remove(img)
    return path


def render_preview(arrays, p, size=512):
    """Render the maps on the chosen shape in a temporary scene; returns PNG path."""
    if p.preview_view == "MAPS":
        return maps_grid_png(arrays, size)
    made = {"images": [], "objects": [], "data": []}
    maps = make_images("PBRGenPreview", arrays, keep=False)
    made["images"] = list(maps.values())
    mat = build_material("PBRGenPreview_Mat", maps, p)
    scene = bpy.data.scenes.new("PBRGenPreview")
    world = bpy.data.worlds.new("PBRGenPreview_World")
    try:
        _set_engine(scene)
        r = scene.render
        r.resolution_x = r.resolution_y = size
        r.resolution_percentage = 100
        r.image_settings.file_format = "PNG"
        r.film_transparent = False

        scene.world = world
        try:
            world.use_nodes = True
        except AttributeError:
            pass
        bg = world.node_tree.nodes.get("Background") if world.node_tree else None
        if bg:
            # Sky-to-ground gradient: gives glossy surfaces something to reflect, so
            # low roughness shows a sharp horizon and high roughness a soft blur.
            wn, wl = world.node_tree.nodes, world.node_tree.links
            tc = wn.new("ShaderNodeTexCoord")
            sep = wn.new("ShaderNodeSeparateXYZ")
            rng_node = wn.new("ShaderNodeMapRange")
            rng_node.inputs["From Min"].default_value = -0.05
            rng_node.inputs["From Max"].default_value = 0.35
            mix = wn.new("ShaderNodeMix")
            mix.data_type = "RGBA"
            _socket(mix.inputs, "A_Color", 6).default_value = (0.03, 0.03, 0.035, 1.0)
            _socket(mix.inputs, "B_Color", 7).default_value = (0.75, 0.78, 0.85, 1.0)
            wl.new(tc.outputs["Generated"], sep.inputs["Vector"])
            wl.new(sep.outputs["Z"], rng_node.inputs["Value"])
            wl.new(rng_node.outputs["Result"], mix.inputs[0])
            wl.new(_socket(mix.outputs, "Result_Color", 2), bg.inputs["Color"])
            bg.inputs["Strength"].default_value = 0.9
        else:
            world.color = (0.18, 0.19, 0.21)

        def add(name, data, loc):
            obj = bpy.data.objects.new(name, data)
            obj.location = loc
            scene.collection.objects.link(obj)
            made["objects"].append(obj)
            made["data"].append(data)
            return obj

        mesh = make_shape_mesh(p.preview_shape)
        mesh.materials.append(mat)
        shape_obj = add("PBRGenPreview_Obj", mesh, (0, 0, 0))
        if p.preview_shape == "MONKEY":
            mod = shape_obj.modifiers.new("Subdivision", "SUBSURF")
            mod.levels = mod.render_levels = 2

        cam_data = bpy.data.cameras.new("PBRGenPreview_Cam")
        cam_data.lens = 50
        cam_loc = (0.0, -3.0, 3.9) if p.preview_shape == "PLANE" else (2.5, -3.75, 2.05)
        if p.preview_shape == "MONKEY":
            cam_loc = (1.4, -4.4, 1.25)
        cam = add("PBRGenPreview_Cam", cam_data, cam_loc)
        _aim(cam)
        scene.camera = cam
        # Close-up zooms in until the surface fills the frame, so detail is easy to judge.
        _frame_object(scene, cam, mesh, fill=1.9 if p.preview_view == "CLOSEUP" else 0.8)

        # Lighting chosen to make setting changes visible:
        #  - a small, low side light rakes across the surface so bumps cast shadows
        #  - a big soft light gives a broad reflection, so roughness changes show
        #  - a weak fill keeps the shadow side readable
        #  - two low "reflection cards" sit where the camera sees their mirror image
        #    in the front and right faces: glossy = sharp bright sheen, rough = none
        for name, energy, loc, sz in (("Key", 900, (-4.0, -2.5, 2.4), 0.8),
                                      ("Soft", 300, (2.0, -3.0, 5.5), 6.0),
                                      ("Fill", 60, (5.0, -1.0, 1.0), 4.0),
                                      ("Rim", 300, (0.5, 5.0, 3.0), 2.0),
                                      ("ReflectFront", 220, (-3.0, -4.5, -2.4), 1.0),
                                      ("ReflectSide", 220, (4.5, 3.0, -2.4), 1.0)):
            ld = bpy.data.lights.new("PBRGenPreview_" + name, "AREA")
            ld.energy = energy
            ld.size = sz
            _aim(add("PBRGenPreview_" + name, ld, loc))

        bpy.ops.render.render(write_still=False, scene=scene.name)
        result = bpy.data.images.get("Render Result")
        if result is None:
            raise RuntimeError("Render produced no image")
        path = _next_preview_path()
        result.save_render(path, scene=scene)
        return path
    finally:
        for obj in made["objects"]:
            bpy.data.objects.remove(obj)
        for data in made["data"]:
            coll = {bpy.types.Mesh: bpy.data.meshes, bpy.types.Camera: bpy.data.cameras}.get(
                type(data), bpy.data.lights)
            coll.remove(data)
        bpy.data.scenes.remove(scene)
        bpy.data.worlds.remove(world)
        bpy.data.materials.remove(mat)
        for img in made["images"]:
            bpy.data.images.remove(img)


def show_preview(path):
    # Image used by the viewport-corner overlay (and as the pixel source below)
    img = bpy.data.images.get(PREVIEW_IMAGE_NAME)
    if img is None:
        img = bpy.data.images.load(path, check_existing=False)
        img.name = PREVIEW_IMAGE_NAME
        # Pixels are already display-ready; stop the GPU converting them to linear (too dark).
        img.colorspace_settings.name = "Non-Color"
    else:
        img.filepath = path
        img.reload()

    # Sidebar preview at full resolution (pcoll.load would shrink it to a 256px icon).
    pcoll = preview_collections["main"]
    pcoll.clear()
    prev = pcoll.new("preview")
    w, h = img.size
    buf = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(buf)
    prev.image_size = (w, h)
    prev.image_pixels_float.foreach_set(buf)

    old = _preview_state["last_path"]
    if old and old != path and os.path.isfile(old):
        try:
            os.remove(old)
        except OSError:
            pass
    _preview_state["last_path"] = path
    _redraw_viewports()


# ---------------------------------------------------------------------------
# Operators (the docstring is the button's hover tooltip)
# ---------------------------------------------------------------------------

_JOB = {"running": False}


def snapshot_settings(p):
    """Plain-Python copy of all settings, safe to read from a worker thread."""
    snap = SimpleNamespace()
    for prop in p.bl_rna.properties:
        ident = prop.identifier
        if ident in ("rna_type", "name") or prop.type == "POINTER":
            continue
        value = getattr(p, ident)
        if prop.type in ("FLOAT", "INT", "BOOLEAN") and getattr(prop, "is_array", False):
            value = tuple(value)
        setattr(snap, ident, value)
    return snap


_EXISTING_ITEMS = [
    ("REPLACE", "Replace It", "Overwrite the existing material and maps. Every object using it updates"),
    ("NEW", "Create New", "Keep the existing one and make a separate new material"),
]


class _ExistingChoice:
    """Shared 'replace or create new?' behaviour for the generate operators."""
    existing: EnumProperty(name="Existing Texture", items=_EXISTING_ITEMS, default="REPLACE",
                           options={"SKIP_SAVE"},
                           description="What to do with the texture that already exists")

    def material_name(self, context):
        raise NotImplementedError

    def use_replace(self, context):
        mode = context.scene.pbrgen.on_existing
        return (self.existing if mode == "ASK" else mode) == "REPLACE"

    def invoke(self, context, event):
        name = self.material_name(context)
        if name and name in bpy.data.materials and context.scene.pbrgen.on_existing == "ASK":
            try:
                return context.window_manager.invoke_props_dialog(
                    self, width=320, title="Texture Already Exists", confirm_text="Generate")
            except TypeError:  # older Blender without title/confirm_text
                return context.window_manager.invoke_props_dialog(self, width=320)
        return self.execute(context)

    def draw(self, context):
        layout = self.layout
        layout.label(text='"%s" already exists.' % self.material_name(context), icon="INFO")
        layout.prop(self, "existing", expand=True)
        layout.label(text="Change this in Output Settings > If It Already Exists")

    # --- background generation -------------------------------------------
    # The heavy number crunching (maps + PNG encoding) runs on a worker thread
    # that never touches Blender data, so Blender stays responsive. Only the
    # final, quick step (creating images and the material) runs on the main
    # thread when the worker is done.

    def start_job(self, context, label, work, done):
        if bpy.app.background or context.window is None:  # scripts / command line: just run it
            done(work(lambda stage: None), context)
            return {"FINISHED"}
        if _JOB["running"]:
            self.report({"WARNING"}, "Already generating a texture; wait for it to finish")
            return {"CANCELLED"}
        state = {"done": False, "result": None, "error": None, "stage": "starting"}

        def run():
            try:
                state["result"] = work(lambda stage: state.__setitem__("stage", stage))
            except Exception as e:  # noqa: BLE001
                state["error"] = e
            finally:
                state["done"] = True

        _JOB["running"] = True
        self._state, self._done, self._label = state, done, label
        self._t0 = time.time()
        threading.Thread(target=run, daemon=True, name="pbrgen-generate").start()
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        self._status(context)
        return {"RUNNING_MODAL"}

    def _status(self, context, clear=False):
        try:
            if clear:
                context.workspace.status_text_set(None)
            else:
                context.workspace.status_text_set(
                    "PBR Maps: generating %s - %s (%.0fs). You can keep working."
                    % (self._label, self._state["stage"], time.time() - self._t0))
        except Exception:  # noqa: BLE001 - status bar is cosmetic
            pass

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if not self._state["done"]:
            self._status(context)
            return {"PASS_THROUGH"}
        context.window_manager.event_timer_remove(self._timer)
        self._status(context, clear=True)
        _JOB["running"] = False
        if self._state["error"] is not None:
            self.report({"ERROR"}, "Failed to generate %s: %s" % (self._label, self._state["error"]))
            return {"CANCELLED"}
        try:
            self._done(self._state["result"], context)
        except Exception as e:  # noqa: BLE001
            self.report({"ERROR"}, "Failed to create %s: %s" % (self._label, e))
            return {"CANCELLED"}
        self.report({"INFO"}, "%s finished in %.1fs" % (self._label, time.time() - self._t0))
        return {"FINISHED"}

    def cancel(self, context):
        _JOB["running"] = False
        self._status(context, clear=True)


class PBRGEN_OT_from_image(_ExistingChoice, bpy.types.Operator):
    """Make Base Color, Height/Bump, Normal, Roughness and AO maps from the chosen image, \
build a material from them and put it on the selected object"""
    bl_idname = "pbrgen.from_image"
    bl_label = "Generate Maps from Image"
    bl_options = {"REGISTER", "UNDO"}

    def material_name(self, context):
        p = context.scene.pbrgen
        if p.source_mode == "FILE":
            src = os.path.basename(bpy.path.abspath(p.filepath))
        elif p.source_mode == "URL":
            src = os.path.basename(urllib.parse.urlparse(p.url.strip()).path) or "download"
        else:
            src = p.source_image.name if p.source_image else ""
        stem = bpy.path.clean_name(os.path.splitext(src)[0]) if src else ""
        return (stem or "Image") + "_PBR" if src else ""

    def execute(self, context):
        p = context.scene.pbrgen
        replace = self.use_replace(context)
        try:
            src = get_source_image(self, p)
        except Exception as e:  # noqa: BLE001 - report anything to the user
            self.report({"ERROR"}, "Could not get image: %s" % e)
            return {"CANCELLED"}
        if src is None:
            return {"CANCELLED"}

        try:
            rgb = read_rgb(src, p.max_size)  # reads Blender image data: main thread
        except Exception as e:  # noqa: BLE001
            self.report({"ERROR"}, "Failed to read image: %s" % e)
            return {"CANCELLED"}
        base = bpy.path.clean_name(os.path.splitext(src.name)[0]) or "Image"
        settings = snapshot_settings(p)
        out_dir = p.out_dir

        def work(stage):
            stage("making maps")
            arrays = image_arrays(rgb, settings)
            stage("encoding images")
            return encode_maps(arrays)

        def done(encoded, context):
            maps = images_from_encoded(base, encoded, out_dir, replace)
            finish(self, context, base + "_PBR", maps, context.scene.pbrgen, replace)

        return self.start_job(context, base, work, done)


class PBRGEN_OT_procedural(_ExistingChoice, bpy.types.Operator):
    """Generate the chosen pattern at full resolution with all PBR maps, \
build a material and put it on the selected object"""
    bl_idname = "pbrgen.procedural"
    bl_label = "Generate Texture"
    bl_options = {"REGISTER", "UNDO"}

    def material_name(self, context):
        return PATTERNS[context.scene.pbrgen.tex_pattern][0].replace(" ", "") + "_PBR"

    def execute(self, context):
        p = context.scene.pbrgen
        replace = self.use_replace(context)
        label = PATTERNS[p.tex_pattern][0]
        base = label.replace(" ", "")
        if p.tex_pattern == "BRICKS" and p.tex_offset > 0 and p.tex_count_y % 2:
            self.report({"WARNING"}, "Odd row count with an offset won't tile seamlessly; use an even number")
        settings = snapshot_settings(p)  # the worker thread must not read Blender data
        size = int(p.tex_res)
        out_dir = p.out_dir

        def work(stage):
            stage("making %dpx maps" % size)
            arrays = procedural_arrays(settings, size)
            stage("encoding images")
            return encode_maps(arrays)

        def done(encoded, context):
            maps = images_from_encoded(base, encoded, out_dir, replace)
            finish(self, context, base + "_PBR", maps, context.scene.pbrgen, replace)

        return self.start_job(context, "%s %dpx" % (label, size), work, done)


class PBRGEN_OT_preview(bpy.types.Operator):
    """Render a quick preview of the material on the chosen shape. \
Nothing is added to your scene"""
    bl_idname = "pbrgen.preview"
    bl_label = "Update Preview"
    bl_options = {"REGISTER"}

    def execute(self, context):
        p = context.scene.pbrgen
        try:
            if p.preview_source == "TEXTURE":
                arrays = procedural_arrays(p, 512)
            else:
                src = get_source_image(self, p)
                if src is None:
                    return {"CANCELLED"}
                arrays = image_arrays(read_rgb(src, min(p.max_size, 1024)), p)
            path = render_preview(arrays, p)
        except Exception as e:  # noqa: BLE001
            self.report({"ERROR"}, "Preview failed: %s" % e)
            return {"CANCELLED"}
        show_preview(path)
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

class PBRGEN_PT_base:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PBR Maps"


class PBRGEN_PT_preview(PBRGEN_PT_base, bpy.types.Panel):
    bl_label = "Preview"
    bl_idname = "PBRGEN_PT_preview"

    def draw(self, context):
        p = context.scene.pbrgen
        sync_viewport_overlay(context)
        layout = self.layout
        layout.row().prop(p, "preview_source", expand=True)

        layout.row().prop(p, "preview_view", expand=True)

        pcoll = preview_collections.get("main")
        if pcoll is not None and "preview" in pcoll:
            ui = context.preferences.system.ui_scale
            layout.template_icon(icon_value=pcoll["preview"].icon_id,
                                 scale=sidebar_icon_scale(context.region.width, ui))
            if p.preview_view == "MAPS":
                grid = layout.grid_flow(columns=2, even_columns=True, align=True)
                for text in ("Top left: Color", "Top right: Normal",
                             "Bottom left: Roughness", "Bottom right: Height"):
                    grid.label(text=text)
        else:
            box = layout.box()
            box.label(text="No preview yet", icon="INFO")
            box.label(text="Press Update Preview below")

        row = layout.row(align=True)
        row.prop(p, "preview_shape", expand=True, icon_only=True)
        layout.label(text="Shape: " + p.bl_rna.properties["preview_shape"].enum_items[p.preview_shape].name)
        layout.row().prop(p, "projection", expand=True)
        row = layout.row(align=True)
        row.operator("pbrgen.preview", icon="FILE_REFRESH")
        row.prop(p, "preview_live", text="Auto", toggle=True, icon="AUTO")
        row = layout.row(align=True)
        row.prop(p, "preview_in_viewport", toggle=True, icon="IMAGE_PLANE")
        sub = row.row(align=True)
        sub.active = p.preview_in_viewport
        sub.prop(p, "preview_viewport_size", text="Size")


class PBRGEN_PT_image(PBRGEN_PT_base, bpy.types.Panel):
    bl_label = "Maps from Image"
    bl_idname = "PBRGEN_PT_image"

    def draw(self, context):
        p = context.scene.pbrgen
        sync_viewport_overlay(context)
        layout = self.layout
        layout.row().prop(p, "source_mode", expand=True)
        if p.source_mode == "FILE":
            layout.prop(p, "filepath", text="")
        elif p.source_mode == "URL":
            layout.prop(p, "url", text="", icon="URL")
        else:
            layout.template_ID(p, "source_image", open="image.open")
        layout.prop(p, "max_size")

        col = layout.column(align=True)
        col.label(text="Height / Bump")
        col.prop(p, "height_blur")
        col.prop(p, "height_contrast")
        col.prop(p, "detail_balance")
        col.prop(p, "invert_height")

        col = layout.column(align=True)
        col.label(text="Color & Roughness")
        col.prop(p, "delight")
        col.prop(p, "rough_min")
        col.prop(p, "rough_max")
        col.prop(p, "invert_roughness")

        layout.operator("pbrgen.from_image", icon="IMAGE_DATA")


class PBRGEN_PT_procedural(PBRGEN_PT_base, bpy.types.Panel):
    bl_label = "Texture Generator"
    bl_idname = "PBRGEN_PT_procedural"

    def draw(self, context):
        p = context.scene.pbrgen
        sync_viewport_overlay(context)
        layout = self.layout
        layout.prop(p, "tex_pattern")
        layout.prop(p, "tex_res")

        fields, labels = PATTERN_UI[p.tex_pattern]
        groups = (
            ("Layout", ["count_x", "count_y", "offset", "gap"]),
            ("Shape", ["bevel", "wear", "bump", "detail", "rivets"]),
            ("Color", ["color1", "color2", "variation", "gap_color"]),
            ("Surface", ["rough", "rough_gap"]),
        )
        for title, keys in groups:
            shown = [k for k in keys if k in fields]
            if not shown:
                continue
            col = layout.column(align=True)
            col.label(text=title)
            for k in shown:
                if k in labels:
                    col.prop(p, "tex_" + k, text=labels[k])
                else:
                    col.prop(p, "tex_" + k)
        layout.prop(p, "tex_seed")
        layout.operator("pbrgen.procedural", icon="TEXTURE")


class PBRGEN_PT_output(PBRGEN_PT_base, bpy.types.Panel):
    bl_label = "Output Settings"
    bl_idname = "PBRGEN_PT_output"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        p = context.scene.pbrgen
        sync_viewport_overlay(context)
        layout = self.layout
        layout.prop(p, "projection")
        if p.projection == "BOX":
            layout.prop(p, "box_blend")
        layout.prop(p, "normal_strength")
        layout.prop(p, "directx_normal")
        layout.prop(p, "ao_strength")
        layout.prop(p, "ao_radius")
        layout.separator()
        layout.prop(p, "tiling")
        layout.prop(p, "ao_mix")
        layout.prop(p, "use_displacement")
        if p.use_displacement:
            layout.prop(p, "displacement_scale")
        layout.prop(p, "assign_to_active")
        layout.prop(p, "on_existing")
        layout.prop(p, "out_dir")


classes = (
    PBRGEN_Props,
    PBRGEN_OT_from_image,
    PBRGEN_OT_procedural,
    PBRGEN_OT_preview,
    PBRGEN_PT_preview,
    PBRGEN_PT_image,
    PBRGEN_PT_procedural,
    PBRGEN_PT_output,
)


def register():
    import bpy.utils.previews
    preview_collections["main"] = bpy.utils.previews.new()
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.pbrgen = PointerProperty(type=PBRGEN_Props)
    if not bpy.app.background:
        _draw_state["handle"] = bpy.types.SpaceView3D.draw_handler_add(
            _draw_viewport_preview, (), "WINDOW", "POST_PIXEL")


def unregister():
    import bpy.utils.previews
    if _draw_state["handle"] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_state["handle"], "WINDOW")
        _draw_state["handle"] = None
    img = bpy.data.images.get(PREVIEW_IMAGE_NAME)
    if img is not None:
        bpy.data.images.remove(img)
    if bpy.app.timers.is_registered(_run_preview_timer):
        bpy.app.timers.unregister(_run_preview_timer)
    del bpy.types.Scene.pbrgen
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    for pcoll in preview_collections.values():
        bpy.utils.previews.remove(pcoll)
    preview_collections.clear()


if __name__ == "__main__":
    register()
