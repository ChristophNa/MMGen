# TPMS Generator Documentation

This document explains how MMGen builds TPMS lattices, with enough theory for non-specialists and implementation details that map directly to the code.

## 1) Quick TPMS Theory (Non-Expert Friendly)

A TPMS (Triply Periodic Minimal Surface) is a repeating 3D pattern described by an implicit function:

- `f(x, y, z) = 0` describes a surface.
- `f(x, y, z) < 0` and `> 0` are the two sides of that surface.

In this project, we use "double" TPMS-style scalar fields and a thickness term:

- `V(x, y, z) = f(x, y, z) - t(x, y, z)^2`
- The extracted surface is the `V = 0` isosurface.
- Intuition: `t` controls local wall thickness. Larger `t` generally makes the solid phase thicker.

Coordinates inside each unit cell are normalized by cell size `a`:

- `X = 2*pi*x/a`, `Y = 2*pi*y/a`, `Z = 2*pi*z/a`.

## 2) Supported TPMS Families

Implemented in `mmgen/tpms_types.py`:

- `GYROID`
- `SCHWARZ_P`
- `DIAMOND`
- `LIDINOID`
- `SPLIT_P`
- `NEOVIUS`

Each has a dedicated equation function registered in `TPMS_REGISTRY`.

## 3) Generation Pipeline (What the Code Actually Does)

Implemented mainly in `mmgen/generator.py`:

1. Build a voxel grid over domain + margin (`generate_grid`).
2. Evaluate TPMS field and thickness grading (`evaluate_field`).
3. Force boundary voxels to positive to encourage closed extraction (`apply_boundary_conditions`).
4. Run marching cubes at iso-level `0` (`generate_raw_mesh`).
5. Shift TPMS mesh from `[0..L, 0..W, 0..H]` to centered coordinates `[-L/2..L/2, ...]`.
6. Optionally add face lids via manifold union.
7. Resolve target geometry (from mesh arg, file path, or domain box fallback).
8. Optionally clip target geometry to the domain box.
9. Intersect TPMS(+lids) with target geometry.
10. Validate quality and return `(mesh, metadata)`.

## 4) Configuration Reference

Defined in `mmgen/config.py`.

### `GenerationConfig`
Top-level config with:

- `lattice: LatticeConfig`
- `sampling: SamplingConfig`
- `booleans: BooleanConfig`
- `geometry: GeometryConfig`

All config models use `extra="forbid"` (unknown keys are rejected).

### `LatticeConfig`

- `type`: TPMS family enum (`GYROID`, `SCHWARZ_P`, etc.); string input is case-insensitive by enum name.
- `cell_size` (`float`, `> 0`): unit-cell size in mm.

### `SamplingConfig`

- `voxels_per_cell` (`int`, `>= 2`): mesh detail per cell axis.
- `margin_cells` (`float`, `>= 0`, default `0.5`): extra sampling margin around domain to improve boundary robustness.

### `GeometryConfig`

- `domain: DomainConfig`
- `lids: LidSpec`
- `thickness: float | GradingSpec`

`DomainConfig`:

- `length`, `width`, `height` in mm (`> 0`).

`LidSpec` (all `>= 0`, `0` disables):

- `x_min`, `x_max`, `y_min`, `y_max`, `z_min`, `z_max`

### `BooleanConfig`

- `lid_overlap_margin` (`float`, default `1.0`): extra lid overlap for robust unions.
- `center_target_mesh` (`bool`, default `True`): center imported target at origin before booleans.
- `clip_target_to_domain` (`bool`, default `True`): intersect target with domain box before final TPMS intersection.

### `GradingSpec`

Implemented in `mmgen/grading.py`.

- `kind="constant"`: `params={"t": ...}`
- `kind="affine"`: `t = a + bx*x + by*y + bz*z`, optional `tmin/tmax` clamp.
- `kind="radial"`: interpolate from `t_center` to `t_outer` up to `radius` around `center`.

## 5) `target_geometry` Behavior

`TPMSGenerator` takes one of:

- `target_mesh=<trimesh.Trimesh>` (in-memory mesh), or
- `target_geometry_path=<Path>` (file loaded with `trimesh.load_mesh`), or
- neither: fallback target is a centered box with domain extents.

Important rules:

- You cannot provide both `target_mesh` and `target_geometry_path` (raises `ValueError`).
- If file loading returns a `trimesh.Scene`, all geometries are concatenated.
- If `center_target_mesh=True`, target centroid is shifted to `[0, 0, 0]`.
- If `clip_target_to_domain=True`, target is clipped by a centered domain box before final intersection.

Practical interpretation:

- `center_target_mesh=True` is usually what you want for "fit this TPMS into this shape" workflows.
- Set `center_target_mesh=False` only if your target mesh is already in the same global coordinates as TPMS.
- Disable `clip_target_to_domain` when you intentionally want geometry outside the configured domain to participate.

## 6) Internal Fixed Extraction Defaults

These are currently not exposed as user parameters:

- Marching-cubes iso-level is fixed to `0`.
- Boundary closure writes `1.0` on outer voxel faces.
- Marching cubes uses `skimage.measure.marching_cubes` defaults (except explicit `spacing`).

## 7) Dependencies for Booleans

Boolean union/intersection use Manifold3D wrappers in `mmgen/utils.py`:

- `mesh_union(...)`
- `mesh_intersection(...)`

If `manifold3d` is missing, boolean operations raise an import error with install guidance.
