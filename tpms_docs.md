# TPMS Generator Documentation

This document describes the generalized implementation of Triply Periodic Minimal Surfaces (TPMS) generation and intersection.

## Architecture

The implementation is split into a modular core:
- `mmgen/tpms_types.py`: Definitions of TPMS equations and registry.
- `mmgen/config.py`: Pydantic models for handling input parameters.
- `mmgen/generator.py`: Main logic for grid generation, field evaluation, and meshing.
- `mmgen/utils.py`: Helper functions (Manifold3D boolean wrappers).

## Supported TPMS Types

The generator supports the following "Double" TPMS variants:

| Type | Equation Summary ($f(x,y,z)$ where $a$ is cell size) |
|---|---|
| **Gyroid** | $sin^2(X)cos^2(Y) + sin^2(Y)cos^2(Z) + sin^2(Z)cos^2(X) + 2(sin(X)cos(Y)sin(Y)cos(Z) + \dots)$ |
| **Schwarz P** | $cos^2(X) + cos^2(Y) + cos^2(Z) + 2(cos(X)cos(Y) + \dots)$ |
| **Diamond** | Mixture of $sin^2$ and $cos^2$ terms |
| **Lidinoid** | Higher frequency terms involving $2X, 2Y, 2Z$ |
| **Split P** | Derived from Schwarz P with frequency doubling |
| **Neovius** | Complex mixture of $cos$ and triple-product terms |

*Note: $X = 2\pi x / a, Y = 2\pi y / a, Z = 2\pi z / a$.*

## Parameters

### GenerationConfig
Top-level deterministic input object for generation.

### LatticeConfig
- `type`: One of `GYROID, SCHWARZ_P, DIAMOND, LIDINOID, SPLIT_P, NEOVIUS`.
- `cell_size`: Size of the unit cell in mm.

### SamplingConfig
- `voxels_per_cell`: Voxel resolution per unit cell.
- `margin_cells`: Grid sampling margin in units of cell size. Default: `0.5`.

### BooleanConfig
- `lid_overlap_margin`: Overlap margin used when constructing lid solids. Default: `1.0`.
- `center_target_mesh`: If `true`, target mesh is recentered to origin before booleans.
- `clip_target_to_domain`: If `true`, target geometry is clipped to domain box before final intersection.

### GeometryConfig
- `domain`: `DomainConfig` (`length`, `width`, `height` in mm).
- `lids`: `LidSpec` per-face lid thicknesses (`x_min`, `x_max`, `y_min`, `y_max`, `z_min`, `z_max`).
- `thickness`: Constant thickness (`float`) or declarative `GradingSpec`.

### GradingSpec
Grading is specified via a declarative `GradingSpec` model (`constant`, `affine`, `radial`).
The field is evaluated as: $V = f(x,y,z) - t(x,y,z)^2$.

### Fixed Internal Extraction Defaults
These remain intentionally internal and are not user-configurable:
- Iso level is fixed at `0`.
- Boundary closure behavior is always applied with fill value `1.0`.
- `skimage.measure.marching_cubes` uses the current built-in defaults.

## Intersection with Manifold3D

The generator uses Manifold3D for boolean operations to ensure clean boundaries at the domain edges and robust handling of non-manifold cases common in marching-cubes output.
