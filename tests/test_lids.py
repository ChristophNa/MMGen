import numpy as np
import trimesh

from mmgen.config import DomainConfig, GenerationConfig, GeometryConfig, LatticeConfig, SamplingConfig
from mmgen.generator import TPMSGenerator
from mmgen.tpms_types import TPMSType


def _build_config(*, lids: dict[str, float] | None = None) -> GenerationConfig:
    return GenerationConfig(
        lattice=LatticeConfig(type=TPMSType.GYROID, cell_size=10.0),
        sampling=SamplingConfig(voxels_per_cell=20),
        geometry=GeometryConfig(
            domain=DomainConfig(length=20, width=20, height=20),
            lids=lids or {},
            thickness=0.5,
        ),
    )


def _verts_on_plane(mesh: trimesh.Trimesh, *, axis: int, value: float, tolerance: float = 0.5) -> int:
    on_plane = np.abs(mesh.vertices[:, axis] - value) < tolerance
    return int(np.sum(on_plane))


def test_z_min_lid_adds_coverage_near_bottom_plane():
    no_lid_mesh, _ = TPMSGenerator(_build_config(lids={})).generate_mesh(allow_nonwatertight=True)
    z_lid_mesh, _ = TPMSGenerator(_build_config(lids={"z_min": 2.0})).generate_mesh(allow_nonwatertight=True)

    lid_count = _verts_on_plane(z_lid_mesh, axis=2, value=-10.0)

    assert lid_count > 0
    assert z_lid_mesh.volume > no_lid_mesh.volume


def test_x_min_x_max_lids_add_coverage_on_both_side_planes():
    no_lid_mesh, _ = TPMSGenerator(_build_config(lids={})).generate_mesh(allow_nonwatertight=True)
    dual_lid_mesh, _ = TPMSGenerator(
        _build_config(lids={"x_min": 2.0, "x_max": 2.0})
    ).generate_mesh(allow_nonwatertight=True)

    lid_min = _verts_on_plane(dual_lid_mesh, axis=0, value=-10.0)
    lid_max = _verts_on_plane(dual_lid_mesh, axis=0, value=10.0)

    assert lid_min > 0
    assert lid_max > 0
    assert dual_lid_mesh.volume > no_lid_mesh.volume
