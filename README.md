# threemf4agent

3MF CLIs for inspecting and editing 3D print models from the shell.

One command, four subcommands — chain them in the shell:

```console
$ threemf inspect part.3mf
unit: Millimeter  meshes: 1
  #0 id=4 verts=8 tris=12  bbox=[0,0,0]..[50,20,30]  size=50×20×30mm
outbox: [0,0,0]..[50,20,30]
```

## `threemf inspect PATH`

List all mesh objects: resource id, name, vertex/triangle counts, bounding box, unit, and reader warnings. Run this first to discover what's in a file.

## `threemf render PATH`

ASCII top-down render (XY plane) of a mesh, shaded by Z height (`low=space, high=@`).

Options:

- `--mesh N` — mesh index (default 0)
- `--width N` — ASCII width in chars, 10..200 (default 70)
- `--focus X0,Y0,X1,Y1` — zoom into an XY window in mm

```console
$ threemf render part.3mf --focus 10,5,30,15
```

## `threemf slice PATH Z`

ASCII horizontal slice through a mesh at Z height (mm, object space) — `#` = wall. Useful for cross-sections, wall thickness, and infill patterns. Same options as `render`.

```console
$ threemf slice part.3mf 15
```

Find the z-range from `threemf inspect` first.

## `threemf modify IN OUT`

Run Python code over a mesh's vertex (`V`, nx3 float) and triangle (`T`, mx3 int) arrays, then save the result as a new 3MF. Code is read from **stdin**; for one-liners pass `--code` instead. In scope: `V`, `T`, `np`, `trimesh`, `mesh` (lib3mf mesh object), `model` (lib3mf model). Leave the modified arrays under the names `V` and `T`. Other meshes and attachments are preserved.

```console
$ cat fix.py | threemf modify in.3mf out.3mf
# one-liner:
$ threemf modify in.3mf out.3mf --code 'V[:, 2] += 5  # lift 5mm'
```

⚠️ This executes arbitrary Python — only run code you trust.

## Install

```console
pip install threemf4agent
```

Python >= 3.10. Built on [lib3mf](https://github.com/3MFConsortium/lib3MF), numpy, and trimesh. Meshes are read in **object space**; build-item transforms are not applied.

## License

MIT — see [LICENSE](LICENSE).
