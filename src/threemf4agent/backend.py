"""Core 3MF operations for the threemf CLI (lib3mf + numpy + trimesh).

Meshes are read in *object space*; build-item transforms are not applied
(ponytail: apply component/build transforms if world-space slices or
plate-relative heights are needed — none of the user's scripts needed them).
"""
import math
import os
import struct
import tempfile
import zlib

import numpy as np
import lib3mf.Lib3MF as L


# ── lib3mf helpers ────────────────────────────────────────────────────────

def load_model(path):
    wrapper = L.Wrapper()
    model = wrapper.CreateModel()
    reader = model.QueryReader("3mf")
    reader.ReadFromFile(path)
    return model, reader


def mesh_objects(model):
    out = []
    it = model.GetMeshObjects()
    while it.MoveNext():
        out.append(it.GetCurrent())
    return out


def mesh_to_arrays(m):
    vs = m.GetVertices()
    V = np.array([[v.Coordinates[i] for i in range(3)] for v in vs], dtype=float)
    T = np.array(
        [[t.Indices[i] for i in range(3)] for t in [m.GetTriangle(i) for i in range(m.GetTriangleCount())]],
        dtype=np.int64,
    )
    return V, T


def write_mesh_to_model(model, m, V, T):
    pos_list = [L.Position((float(x), float(y), float(z))) for x, y, z in V]
    tri_list = [L.Triangle((int(a), int(b), int(c))) for a, b, c in T]
    m.SetGeometry(pos_list, tri_list)


# ── commands ──────────────────────────────────────────────────────────────

def cmd_inspect(path):
    model, reader = load_model(path)
    meshes = mesh_objects(model)
    out = {
        "unit": str(model.GetUnit()),
        "mesh_count": len(meshes),
        "meshes": [],
    }
    for i, m in enumerate(meshes):
        V, T = mesh_to_arrays(m)
        entry = {
            "index": i,
            "resource_id": m.GetResourceID(),
            "name": m.GetName() if hasattr(m, "GetName") else "",
            "vertices": int(len(V)),
            "triangles": int(len(T)),
        }
        if len(V):
            mn, mx = V.min(axis=0), V.max(axis=0)
            entry["bbox_min"] = [round(float(v), 3) for v in mn]
            entry["bbox_max"] = [round(float(v), 3) for v in mx]
            entry["size"] = [round(float(mx[k] - mn[k]), 3) for k in range(3)]
        out["meshes"].append(entry)
    # outbox (lib3mf leaves ±FLT_MAX when the file has no build)
    try:
        ob = model.GetOutbox()
        mn = [float(ob.MinCoordinate[i]) for i in range(3)]
        mx = [float(ob.MaxCoordinate[i]) for i in range(3)]
        if max(map(abs, mn + mx)) < 1e37:
            out["outbox_min"] = [round(v, 3) for v in mn]
            out["outbox_max"] = [round(v, 3) for v in mx]
    except Exception:
        pass
    wc = reader.GetWarningCount()
    if wc:
        out["warnings"] = [reader.GetWarning(i)[1] for i in range(wc)]
    return out


def _pick_mesh(model, idx):
    meshes = mesh_objects(model)
    if not meshes:
        raise ValueError("no mesh objects in file")
    if idx < 0 or idx >= len(meshes):
        raise ValueError(f"mesh index {idx} out of range (0..{len(meshes)-1})")
    return meshes[idx], idx


def _rotate(V, elev=90.0, azim=0.0):
    """Camera rotation.  elev=90: top view (looking down Z).  elev=0: side view
    (looking along Y, +Z up).  azim: turntable rotation around Z in degrees.
    Returns (Vr, zs) where Vr[:,0]=screen-x, Vr[:,1]=screen-y(+up), Vr[:,2]=depth
    (+closer to camera), and zs = original Z (for height-based shading)."""
    az = math.radians(azim)
    ca, sa = math.cos(az), math.sin(az)
    x = V[:, 0] * ca - V[:, 1] * sa          # azimuth around Z
    y = V[:, 0] * sa + V[:, 1] * ca
    z = V[:, 2].copy()
    el = math.radians(elev)
    ce, se = math.cos(el), math.sin(el)
    sy = y * se + z * ce                      # screen-y (+up)
    dp = y * ce + z * se                      # depth (+closer)
    Vr = np.column_stack([x, sy, dp])
    return Vr, z


def cmd_render(path, mesh=0, width=70, focus=None, elev=90.0, azim=0.0):
    model, _ = load_model(path)
    m, idx = _pick_mesh(model, mesh)
    V, T = mesh_to_arrays(m)
    if len(V) == 0:
        return {"ascii": "(empty mesh)", "mesh_index": idx}
    Vr, zs = _rotate(V, elev, azim)
    art = render_top(Vr, T, width, focus=focus, shade=zs, axes=("sx", "sy", "Z"))
    mn, mx = V.min(axis=0), V.max(axis=0)
    el = int(round(elev))
    view = f"elev={el}° azim={int(round(azim))}°, shaded by height Z"
    return {
        "mesh_index": idx,
        "view": view,
        "size_mm": [round(float(mx[k] - mn[k]), 3) for k in range(3)],
        "ramp": "low Z -> high Z:  ' .:-=+*#%@",
        "ascii": art,
        "png": _write_temp_png(render_top_png(Vr, T, focus=focus, shade=zs)),
    }


def cmd_slice(path, z, mesh=0, width=70, focus=None):
    model, _ = load_model(path)
    m, idx = _pick_mesh(model, mesh)
    V, T = mesh_to_arrays(m)
    if len(V) == 0:
        return {"ascii": "(empty mesh)", "mesh_index": idx}
    mn, mx = V.min(axis=0), V.max(axis=0)
    if z < float(mn[2]) - 1e-6 or z > float(mx[2]) + 1e-6:
        return {
            "mesh_index": idx,
            "ascii": f"(z={z:.3f} outside mesh z-range {mn[2]:.3f}..{mx[2]:.3f})",
            "segment_count": 0,
        }
    segs = slice_segments(V, T, z)
    art = render_segments(segs, V, width, focus=focus)
    return {
        "mesh_index": idx,
        "view": f"horizontal slice at z={z:.3f} mm (XY plane)",
        "segment_count": len(segs),
        "ascii": art,
        "png": _write_temp_png(render_slice_png(segs, focus=focus)),
    }


def cmd_modify(path, out_path, code, mesh=0):
    model, _ = load_model(path)
    m, idx = _pick_mesh(model, mesh)
    V, T = mesh_to_arrays(m)
    globs = {
        "V": V.copy(),
        "T": T.copy(),
        "np": np,
        "trimesh": __import__("trimesh"),
        "mesh": m,
        "model": model,
    }
    exec(compile(code, "<modify_3mf>", "exec"), globs)
    Vn = globs["V"]
    Tn = globs["T"]
    Vn = np.asarray(Vn, dtype=float).reshape(-1, 3)
    Tn = np.asarray(Tn, dtype=np.int64).reshape(-1, 3)
    if len(Tn) and Tn.max() >= len(Vn):
        raise ValueError(
            f"triangle indices out of range: max index {int(Tn.max())} >= vertex count {len(Vn)}"
        )
    write_mesh_to_model(model, m, Vn, Tn)
    writer = model.QueryWriter("3mf")
    writer.WriteToFile(out_path)
    mn, mx = Vn.min(axis=0), Vn.max(axis=0)
    return {
        "mesh_index": idx,
        "out_path": out_path,
        "vertices": int(len(Vn)),
        "triangles": int(len(Tn)),
        "bbox_min": [round(float(v), 3) for v in mn],
        "bbox_max": [round(float(v), 3) for v in mx],
        "note": "mesh written in object space; other meshes/attachments preserved",
    }


# ── ASCII renderers ───────────────────────────────────────────────────────

_RAMP = " .:-=+*#%@"

# ── PNG output (stdlib zlib/struct, no extra deps) ────────────────────────

_PNG_SIZE = 800  # px, long edge
_PNG_BG = (18, 18, 26)
_Z_STOPS = np.array([0.0, 0.35, 0.6, 0.85, 1.0])
_Z_RAMP = np.array(
    [[16, 20, 48], [40, 90, 170], [50, 175, 175], [235, 225, 130], [250, 250, 250]], float
)


def _png_chunk(tag, data):
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_png(path, rgb):
    """(H,W,3) uint8 array -> 8-bit RGB PNG (unfiltered rows)."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    payload = (b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr)
               + _png_chunk(b"IDAT", zlib.compress(raw, 6)) + _png_chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(payload)


def render_top_png(V, T, size=_PNG_SIZE, focus=None, shade=None):
    """Software z-buffer render.  V must be pre-rotated (col0=screen-x, col1=screen-y,
    col2=depth).  shade = per-vertex shading values (default: V[:,2] = depth).
    Returns (H,W,3) uint8."""
    zs = shade if shade is not None else V[:, 2]
    fb = _focus_bounds(focus)
    xs, ys = V[:, 0], V[:, 1]
    if fb:
        x0, y0, x1, y1 = fb
    else:
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
    dx = (x1 - x0) or 1.0
    dy = (y1 - y0) or 1.0
    W = size
    H = max(1, min(1600, int(round(size * dy / dx))))
    P = np.empty((len(V), 2), float)
    P[:, 0] = (V[:, 0] - x0) / dx * (W - 1)
    P[:, 1] = (y1 - V[:, 1]) / dy * (H - 1)  # +screen-y up
    dbuf = np.full((H, W), -np.inf)  # depth, for occlusion
    zbuf = np.full((H, W), -np.inf)  # shade value of frontmost face
    nbuf = np.zeros((H, W, 3), float)  # surface normal of frontmost face
    zt = V[:, 2]
    # per-triangle face normals (flat shading), flipped to face the camera
    _vn = V[T]
    Ntri = np.cross(_vn[:, 1] - _vn[:, 0], _vn[:, 2] - _vn[:, 0])
    Ntri /= np.linalg.norm(Ntri, axis=1, keepdims=True) + 1e-30
    Ntri[Ntri[:, 2] < 0] *= -1
    for (a, b, c), zt3, z3, nrm in zip(P[T], zt[T], zs[T], Ntri):
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(d) < 1e-12:
            continue
        gx0 = max(0, int(math.floor(min(a[0], b[0], c[0]))))
        gx1 = min(W - 1, int(math.ceil(max(a[0], b[0], c[0]))))
        gy0 = max(0, int(math.floor(min(a[1], b[1], c[1]))))
        gy1 = min(H - 1, int(math.ceil(max(a[1], b[1], c[1]))))
        if gx0 > gx1 or gy0 > gy1:
            continue
        gy, gx = np.mgrid[gy0:gy1 + 1, gx0:gx1 + 1]
        w0 = ((b[1] - c[1]) * (gx - c[0]) + (c[0] - b[0]) * (gy - c[1])) / d
        w1 = ((c[1] - a[1]) * (gx - c[0]) + (a[0] - c[0]) * (gy - c[1])) / d
        m = (w0 >= 0) & (w1 >= 0) & (w0 + w1 <= 1.0)
        if not m.any():
            continue
        dd = (w0 * zt3[0] + w1 * zt3[1] + (1.0 - w0 - w1) * zt3[2])[m]  # depth test
        zz = (w0 * z3[0] + w1 * z3[1] + (1.0 - w0 - w1) * z3[2])[m]    # height for color
        rows, cols = gy[m], gx[m]
        up = dd > dbuf[rows, cols]
        if up.any():
            dbuf[rows[up], cols[up]] = dd[up]
            zbuf[rows[up], cols[up]] = zz[up]
            nbuf[rows[up], cols[up]] = nrm
    zmin, zmax = float(zs.min()), float(zs.max())
    t = np.clip((zbuf - zmin) / ((zmax - zmin) or 1.0), 0, 1)
    rgb = np.empty((H, W, 3), np.uint8)
    rgb[...] = _PNG_BG
    vis = np.isfinite(zbuf)
    base = np.stack([np.interp(t[vis], _Z_STOPS, _Z_RAMP[:, k]) for k in range(3)], 1)
    L = np.array([0.35, 0.35, 1.0])
    L /= np.linalg.norm(L)
    diff = np.clip(nbuf[vis] @ L, 0, 1)
    intensity = 0.25 + 0.75 * diff
    rgb[vis] = (base * intensity[:, None]).round().clip(0, 255).astype(np.uint8)
    return rgb


def render_slice_png(segs, size=_PNG_SIZE, focus=None):
    """Rasterize slice segments straight from the mesh intersections, 2x2-supersampled
    for antialiasing. Returns (H,W,3) uint8 or None if there is nothing to draw."""
    if not segs:
        return None
    fb = _focus_bounds(focus)
    if fb:
        x0, y0, x1, y1 = fb
        segs = [s for s in segs if not (
            max(s[0][0], s[1][0]) < x0 or min(s[0][0], s[1][0]) > x1
            or max(s[0][1], s[1][1]) < y0 or min(s[0][1], s[1][1]) > y1
        )]
        if not segs:
            return None
    else:
        allp = [p for s in segs for p in s]
        x0, x1 = min(p[0] for p in allp), max(p[0] for p in allp)
        y0, y1 = min(p[1] for p in allp), max(p[1] for p in allp)
    # pad so boundary walls don't sit on the image edge (half-clipped stroke)
    x0, x1 = x0 - 0.01 * (x1 - x0), x1 + 0.01 * (x1 - x0)
    y0, y1 = y0 - 0.01 * (y1 - y0), y1 + 0.01 * (y1 - y0)
    dx = (x1 - x0) or 1.0
    dy = (y1 - y0) or 1.0
    W = size
    H = max(1, min(1600, int(round(size * dy / dx))))
    SS = 2  # ponytail: 2x2 supersample; bump to 3x3 if thin walls look faint
    H2, W2 = H * SS, W * SS
    canvas = np.zeros((H2, W2), bool)
    stamps = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    for (p, q) in segs:
        ax = (p[0] - x0) / dx * (W - 1) * SS
        ay = (p[1] - y0) / dy * (H - 1) * SS
        bx = (q[0] - x0) / dx * (W - 1) * SS
        by = (q[1] - y0) / dy * (H - 1) * SS
        n = int(4 * max(abs(bx - ax), abs(by - ay))) + 1  # dense: 4 samples per SS cell
        t = np.linspace(0.0, 1.0, n)
        xs = ax + t * (bx - ax)
        ys = ay + t * (by - ay)
        for dxs, dys in stamps:
            xi = np.clip(np.round(xs + dxs).astype(int), 0, W2 - 1)
            yi = np.clip(np.round(ys + dys).astype(int), 0, H2 - 1)
            canvas[yi, xi] = True
    cov = canvas.reshape(H, SS, W, SS).sum(axis=(1, 3)) / (SS * SS)
    bg = np.asarray(_PNG_BG, float)
    wall = np.asarray([240, 240, 245], float)
    rgb = bg[None, None, :] * (1 - cov[:, :, None]) + wall[None, None, :] * cov[:, :, None]
    return rgb.round().astype(np.uint8)


def _write_temp_png(rgb):
    if rgb is None:
        return None
    fd, path = tempfile.mkstemp(prefix="threemf_", suffix=".png")
    os.close(fd)
    write_png(path, rgb)
    return path


def _focus_bounds(focus):
    """focus = [xmin, ymin, xmax, ymax] in mm, or None."""
    if not focus or len(focus) != 4:
        return None
    fx0, fy0, fx1, fy1 = (float(v) for v in focus)
    if fx0 > fx1:
        fx0, fx1 = fx1, fx0
    if fy0 > fy1:
        fy0, fy1 = fy1, fy0
    return fx0, fy0, fx1, fy1


def _tri_xy_overlap(V, tri, x0, y0, x1, y1):
    xs = [V[tri[0]][0], V[tri[1]][0], V[tri[2]][0]]
    ys = [V[tri[0]][1], V[tri[1]][1], V[tri[2]][1]]
    return not (max(xs) < x0 or min(xs) > x1 or max(ys) < y0 or min(ys) > y1)


def render_top(V, T, width, focus=None, axes=("u", "v", "w"), shade=None):
    fb = _focus_bounds(focus)
    zs = shade if shade is not None else V[:, 2]
    xs, ys = V[:, 0], V[:, 1]
    if fb:
        x0, y0, x1, y1 = fb
    else:
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
    dx = (x1 - x0) or 1.0
    dy = (y1 - y0) or 1.0
    # char cells are ~2x taller than wide -> scale rows by 0.5
    height = max(1, int(round(width * (dy / dx) * 0.5)))
    height = min(height, 60)
    zmin, zmax = float(zs.min()), float(zs.max())
    dz = (zmax - zmin) or 1.0
    grid = [[None] * width for _ in range(height)]
    gx_scale = (width - 1) / dx
    gy_scale = (height - 1) / dy
    vis_tris = [t for t in T if _tri_xy_overlap(V, t, x0, y0, x1, y1)] if len(T) > 4096 else T
    if fb is not None and len(vis_tris):
        vz = zs[vis_tris]
        zmin, zmax = float(vz.min()), float(vz.max())
        dz = (zmax - zmin) or 1.0
    for tri in vis_tris:
        a, b, c = V[tri[0]], V[tri[1]], V[tri[2]]
        px = [(a[0] - x0) * gx_scale, (b[0] - x0) * gx_scale, (c[0] - x0) * gx_scale]
        py = [(a[1] - y0) * gy_scale, (b[1] - y0) * gy_scale, (c[1] - y0) * gy_scale]
        zvals = [zs[tri[0]], zs[tri[1]], zs[tri[2]]]
        minx = int(math.floor(min(px)))
        maxx = int(math.ceil(max(px)))
        miny = int(math.floor(min(py)))
        maxy = int(math.ceil(max(py)))
        for gy in range(max(0, miny), min(height, maxy + 1)):
            for gx in range(max(0, minx), min(width, maxx + 1)):
                # barycentric: containment test and Z interpolation from one pass
                d = (py[1] - py[2]) * (px[0] - px[2]) + (px[2] - px[1]) * (py[0] - py[2])
                if abs(d) < 1e-12:
                    continue
                ba = ((py[1] - py[2]) * (gx - px[2]) + (px[2] - px[1]) * (gy - py[2])) / d
                bb = ((py[2] - py[0]) * (gx - px[2]) + (px[0] - px[2]) * (gy - py[2])) / d
                if ba < 0 or bb < 0 or ba + bb > 1.0:
                    continue
                z = ba * zvals[0] + bb * zvals[1] + (1.0 - ba - bb) * zvals[2]
                cur = grid[gy][gx]
                if cur is None or z > cur:
                    grid[gy][gx] = z
    lines = []
    for gy in range(height - 1, -1, -1):  # +y up
        row = []
        for gx in range(width):
            z = grid[gy][gx]
            if z is None:
                row.append(" ")
            else:
                t = (z - zmin) / dz
                t = 0.0 if t < 0 else (1.0 if t > 1 else t)
                row.append(_RAMP[min(len(_RAMP) - 1, int(t * (len(_RAMP) - 1)))])
        lines.append("".join(row))
    # axis caption
    lines.append(f"{axes[0]}: {x0:.2f} -> {x1:.2f} mm  (left -> right){'  [zoomed]' if fb else ''}")
    lines.append(f"{axes[1]}: {y0:.2f} -> {y1:.2f} mm  (bottom -> top){'  [zoomed]' if fb else ''}")
    lines.append(f"{axes[2]}: {zmin:.2f} -> {zmax:.2f} mm  (shading{' in view' if fb else ''})")
    return "\n".join(lines)


def slice_segments(V, T, z):
    segs = []
    for tri in T:
        a, b, c = V[tri[0]], V[tri[1]], V[tri[2]]
        pts = []
        for p, q in ((a, b), (b, c), (c, a)):
            z0, z1 = float(p[2]), float(q[2])
            if z0 == z1:
                if z0 == z:
                    # edge lies on the plane — keep endpoints (lazy: handles
                    # coplanar faces; may double-count but render stays clean)
                    pts.append((float(p[0]), float(p[1])))
                    pts.append((float(q[0]), float(q[1])))
                continue
            if (z0 - z) * (z1 - z) <= 0:
                t = (z - z0) / (z1 - z0)
                pts.append((float(p[0] + t * (q[0] - p[0])), float(p[1] + t * (q[1] - p[1]))))
        # dedupe
        uniq = []
        for p in pts:
            if not any(abs(p[0] - q[0]) < 1e-7 and abs(p[1] - q[1]) < 1e-7 for q in uniq):
                uniq.append(p)
        if len(uniq) == 2:
            segs.append((uniq[0], uniq[1]))
        elif len(uniq) > 2:
            # degenerate (vertex on plane) — emit as a fan from first point
            for q in uniq[1:]:
                segs.append((uniq[0], q))
    return segs


def render_segments(segs, V, width, focus=None):
    if not segs:
        return "(no intersections at this height)"
    fb = _focus_bounds(focus)
    allp = [p for s in segs for p in s]
    xs = [p[0] for p in allp]
    ys = [p[1] for p in allp]
    if fb:
        x0, y0, x1, y1 = fb
        # cull segments fully outside the focus window
        segs = [s for s in segs if not (
            max(s[0][0], s[1][0]) < x0 or min(s[0][0], s[1][0]) > x1
            or max(s[0][1], s[1][1]) < y0 or min(s[0][1], s[1][1]) > y1
        )]
        if not segs:
            return f"(no intersections in focus window x[{x0:.2f},{x1:.2f}] y[{y0:.2f},{y1:.2f}])"
    else:
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
    dx = (x1 - x0) or 1.0
    dy = (y1 - y0) or 1.0
    height = max(1, int(round(width * (dy / dx) * 0.5)))
    height = min(height, 60)
    gx_scale = (width - 1) / dx
    gy_scale = (height - 1) / dy
    grid = [[" "] * width for _ in range(height)]

    def put(gx_f, gy_f):
        gx = int(round(gx_f))
        gy = int(round(gy_f))
        if 0 <= gx < width and 0 <= gy < height:
            grid[gy][gx] = "#"

    for (p, q) in segs:
        ax = (p[0] - x0) * gx_scale
        ay = (p[1] - y0) * gy_scale
        bx = (q[0] - x0) * gx_scale
        by = (q[1] - y0) * gy_scale
        # rasterize line (DDA)
        steps = int(max(abs(bx - ax), abs(by - ay))) + 1
        for s in range(steps + 1):
            t = s / steps
            put(ax + t * (bx - ax), ay + t * (by - ay))
    lines = []
    for gy in range(height - 1, -1, -1):
        lines.append("".join(grid[gy]))
    lines.append(f"x: {x0:.2f} -> {x1:.2f} mm  (left -> right){'  [zoomed]' if fb else ''}")
    lines.append(f"y: {y0:.2f} -> {y1:.2f} mm  (bottom -> top){'  [zoomed]' if fb else ''}")
    lines.append(f"slice loops approx {len(segs)} segments (# = wall)")
    return "\n".join(lines)



