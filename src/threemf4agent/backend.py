"""Core 3MF operations for the threemf CLI (lib3mf + numpy + trimesh).

Meshes are read in *object space*; build-item transforms are not applied
(ponytail: apply component/build transforms if world-space slices or
plate-relative heights are needed — none of the user's scripts needed them).
"""
import math

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
        "file": path,
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


def cmd_render(path, mesh=0, width=70, focus=None):
    model, _ = load_model(path)
    m, idx = _pick_mesh(model, mesh)
    V, T = mesh_to_arrays(m)
    if len(V) == 0:
        return {"ascii": "(empty mesh)", "mesh_index": idx}
    art = render_top(V, T, width, focus=focus)
    mn, mx = V.min(axis=0), V.max(axis=0)
    return {
        "mesh_index": idx,
        "view": "top (XY), shaded by height Z",
        "size_mm": [round(float(mx[k] - mn[k]), 3) for k in range(3)],
        "ramp": "low Z -> high Z:  ' .:-=+*#%@",
        "ascii": art,
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


def render_top(V, T, width, focus=None):
    fb = _focus_bounds(focus)
    xs, ys, zs = V[:, 0], V[:, 1], V[:, 2]
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
        vz = V[vis_tris][:, :, 2]
        zmin, zmax = float(vz.min()), float(vz.max())
        dz = (zmax - zmin) or 1.0
    for tri in vis_tris:
        a, b, c = V[tri[0]], V[tri[1]], V[tri[2]]
        px = [(a[0] - x0) * gx_scale, (b[0] - x0) * gx_scale, (c[0] - x0) * gx_scale]
        py = [(a[1] - y0) * gy_scale, (b[1] - y0) * gy_scale, (c[1] - y0) * gy_scale]
        zvals = [a[2], b[2], c[2]]
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
    lines.append(f"x: {x0:.2f} -> {x1:.2f} mm  (left -> right){'  [zoomed]' if fb else ''}")
    lines.append(f"y: {y0:.2f} -> {y1:.2f} mm  (bottom -> top){'  [zoomed]' if fb else ''}")
    lines.append(f"z: {zmin:.2f} -> {zmax:.2f} mm  (shading{' in view' if fb else ''})")
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



