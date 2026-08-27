"""threemf — inspect, render, slice, and modify 3MF files from the shell.

Run `threemf help` for the full manual.
"""
import argparse
import os
import sys
from importlib.resources import files

from . import backend


def _abs(path: str, must_exist: bool = True) -> str:
    p = os.path.abspath(path)
    if must_exist and not os.path.exists(p):
        raise FileNotFoundError(f"file not found: {p}")
    return p


def _width(s: str) -> int:
    v = int(s)
    if not 10 <= v <= 200:
        raise argparse.ArgumentTypeError("width must be 10..200")
    return v


def _focus(s: str):
    try:
        vals = [float(v) for v in s.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("expected x0,y0,x1,y1 as comma-separated floats")
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("expected 4 values: x0,y0,x1,y1")
    return vals


def _box(mn, mx) -> str:
    return f"[{','.join(str(v) for v in mn)}]..[{','.join(str(v) for v in mx)}]"


def _view_opts(p) -> None:
    p.add_argument("--mesh", type=int, default=0, help="mesh index (default 0)")
    p.add_argument("--width", type=_width, default=70, help="ASCII width in chars, 10..200 (default 70)")
    p.add_argument("--focus", type=_focus, metavar="X0,Y0,X1,Y1", help="zoom into an XY window in mm")


def cmd_inspect(a) -> None:
    r = backend.cmd_inspect(_abs(a.path))
    print(f"unit: {r['unit']}  meshes: {r['mesh_count']}")
    for m in r["meshes"]:
        bb = ""
        if m.get("bbox_min"):
            bb = f"  bbox={_box(m['bbox_min'], m['bbox_max'])}  size={'×'.join(str(v) for v in m['size'])}mm"
        name = f" name={m['name']}" if m.get("name") else ""
        print(f"  #{m['index']} id={m['resource_id']}{name} verts={m['vertices']} tris={m['triangles']}{bb}")
    if r.get("outbox_min"):
        print(f"outbox: {_box(r['outbox_min'], r['outbox_max'])}")
    if r.get("warnings"):
        print("warnings: " + "; ".join(r["warnings"]))


def _head(r: dict) -> str:
    head = f"mesh #{r['mesh_index']}"
    if r.get("view"):
        head += f" — {r['view']}"
    return head


def cmd_render(a) -> None:
    r = backend.cmd_render(_abs(a.path), mesh=a.mesh, width=a.width, focus=a.focus, elev=a.elev, azim=a.azim)
    print(_head(r))
    if "size_mm" in r:
        print(f"size: {'×'.join(str(v) for v in r['size_mm'])} mm  width: {a.width}")
        print(f"shading: {r['ramp']}")
        print()
    print(r["ascii"])
    if r.get("png"):
        print(f"png: {r['png']}")


def cmd_slice(a) -> None:
    r = backend.cmd_slice(_abs(a.path), z=a.z, mesh=a.mesh, width=a.width, focus=a.focus)
    print(_head(r) + (f"  segments: {r['segment_count']}" if "segment_count" in r else ""))
    print(r["ascii"])
    if r.get("png"):
        print(f"png: {r['png']}")


def cmd_help(a) -> None:
    print(files("threemf4agent").joinpath("help.md").read_text().rstrip())


def cmd_modify(a) -> None:
    r = backend.cmd_modify(_abs(a.path), _abs(a.out_path, must_exist=False), a.code, mesh=a.mesh)
    print(f"✓ wrote {r['out_path']}")
    print(f"mesh #{r['mesh_index']}: {r['vertices']} verts, {r['triangles']} tris")
    if r.get("bbox_min"):
        print(f"bbox: {_box(r['bbox_min'], r['bbox_max'])}")
    print(r["note"])
    print(f"verify with: threemf render {r['out_path']}")


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="threemf",
        description="Inspect, render, slice, and modify 3MF files. Run 'threemf help' for the full manual.",
    )
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("inspect", help="list meshes, counts, bbox, unit, warnings")
    p.add_argument("path", help="path to the .3mf file")
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("render", help="ASCII render from a camera angle, shaded by height Z")
    p.add_argument("path", help="path to the .3mf file")
    _view_opts(p)
    p.add_argument("--elev", type=float, default=90.0, metavar="DEG",
                   help="camera elevation in degrees: 90=top (default), 0=side")
    p.add_argument("--azim", type=float, default=0.0, metavar="DEG",
                   help="camera azimuth (turntable around Z) in degrees (default 0)")
    p.set_defaults(fn=cmd_render)

    p = sub.add_parser("slice", help="ASCII horizontal slice at Z (XY plane, # = wall)")
    p.add_argument("path", help="path to the .3mf file")
    p.add_argument("z", type=float, help="Z height in mm (object space)")
    _view_opts(p)
    p.set_defaults(fn=cmd_slice)

    p = sub.add_parser("modify", help="run Python code over the mesh V/T arrays, save as new 3MF")
    p.add_argument("path", help="input .3mf file")
    p.add_argument("out_path", help="output .3mf file")
    p.add_argument("--mesh", type=int, default=0, help="mesh index to modify (default 0)")
    p.set_defaults(fn=cmd_modify)

    p = sub.add_parser("help", help="show this help", add_help=False)
    p.set_defaults(fn=cmd_help)

    args = ap.parse_args()
    if args.cmd is None:
        cmd_help(args)
        return
    if args.cmd == "modify":
        if sys.stdin.isatty():
            print("✗ code comes from stdin: cat fix.py | threemf modify in.3mf out.3mf", file=sys.stderr)
            sys.exit(1)
        args.code = sys.stdin.read()
    try:
        args.fn(args)
    except Exception as e:
        print(f"✗ {e}", file=sys.stderr)
        sys.exit(1)
