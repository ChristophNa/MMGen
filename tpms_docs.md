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

### TPMSParams
- `type`: One of `GYROID, SCHWARZ_P, DIAMOND, LIDINOID, SPLIT_P, NEOVIUS`.
- `cell_size`: Size of the unit cell in mm.
- `resolution`: Voxel resolution per unit cell (e.g., `30j` for 30 voxels).

### GradingSpec
Grading is specified via a declarative `GradingSpec` model (constant/linear/radial).
The field is evaluated as: $V = f(x,y,z) - t(x,y,z)^2$.

### DomainConfig
- `length, width, height`: Dimensions of the bounding box/domain in mm.

### GeneratorConfig
- Combines the above parameters.
- `target_geometry`: If provided (path to STL), the TPMS will be intersected with this geometry using Manifold3D. If not provided, it intersects with a block defined by the domain.

## Intersection with Manifold3D

The generator uses Manifold3D for boolean operations to ensure clean boundaries at the domain edges and robust handling of non-manifold cases common in marching-cubes output.
