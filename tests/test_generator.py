import hashlib

import numpy as np
import pytest
import trimesh
from pydantic import ValidationError

from mmgen.config import (
    BooleanConfig,
    DomainConfig,
    GenerationConfig,
    GeometryConfig,
    LatticeConfig,
    LidSpec,
    SamplingConfig,
)
from mmgen.generator import MeshQualityMetadata, TPMSGenerator
from mmgen.grading import GradingSpec
from mmgen.tpms_types import TPMSType


def _base_config(
    *,
    tpms_type: TPMSType = TPMSType.GYROID,
    cell_size: float = 10.0,
    voxels_per_cell: int = 20,
    domain: DomainConfig | None = None,
    thickness: float | GradingSpec = 0.5,
    lids: LidSpec | dict[str, float] | None = None,
    margin_cells: float = 0.5,
    booleans: BooleanConfig | None = None,
) -> GenerationConfig:
    return GenerationConfig(
        lattice=LatticeConfig(type=tpms_type, cell_size=cell_size),
        sampling=SamplingConfig(voxels_per_cell=voxels_per_cell, margin_cells=margin_cells),
        booleans=booleans or BooleanConfig(),
        geometry=GeometryConfig(
            domain=domain or DomainConfig(length=20, width=20, height=20),
            thickness=thickness,
            lids=lids or LidSpec(),
        ),
    )


def _mesh_signature(mesh: trimesh.Trimesh) -> str:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    payload = np.round(verts, 8).tobytes() + faces.tobytes()
    return hashlib.sha256(payload).hexdigest()


def test_basic_gyroid(tmp_path):
    output_path = tmp_path / "basic_gyroid.stl"
    config = _base_config(
        tpms_type=TPMSType.GYROID,
        voxels_per_cell=30,
        domain=DomainConfig(length=30, width=30, height=30),
        thickness=0.5,
    )

    gen = TPMSGenerator(config)
    mesh, metadata = gen.generate_mesh(allow_nonwatertight=True)
    written = gen.export(mesh, output_path)

    assert isinstance(mesh, trimesh.Trimesh)
    assert isinstance(metadata, MeshQualityMetadata)
    assert not mesh.is_empty
    assert metadata.triangle_count > 0
    assert isinstance(metadata.warnings, list)
    assert np.isfinite(np.asarray(metadata.bbox)).all()
    assert written == output_path
    assert output_path.exists()


def test_graded_schwarz_p(tmp_path):
    output_path = tmp_path / "graded_schwarz_p.stl"
    grading_spec = GradingSpec(
        kind="affine",
        params={"a": 0.2, "bx": (0.8 - 0.2) / 50.0, "by": 0.0, "bz": 0.0},
    )
    config = _base_config(
        tpms_type=TPMSType.SCHWARZ_P,
        voxels_per_cell=30,
        domain=DomainConfig(length=50, width=20, height=20),
        thickness=grading_spec,
    )

    gen = TPMSGenerator(config)
    mesh, metadata = gen.generate_mesh(allow_nonwatertight=True)
    written = gen.export(mesh, output_path)

    assert isinstance(mesh, trimesh.Trimesh)
    assert isinstance(metadata, MeshQualityMetadata)
    assert not mesh.is_empty
    assert metadata.triangle_count > 0
    assert isinstance(metadata.warnings, list)
    assert np.isfinite(np.asarray(metadata.bbox)).all()
    assert written == output_path
    assert output_path.exists()


def test_lidinoid_with_target(tmp_path):
    output_path = tmp_path / "lidinoid_mesh.stl"

    dummy_target = tmp_path / "dummy_target.stl"
    box = trimesh.primitives.Box(extents=(10, 10, 10))
    box.export(dummy_target)

    config = _base_config(
        tpms_type=TPMSType.LIDINOID,
        cell_size=5.0,
        voxels_per_cell=20,
        domain=DomainConfig(length=20, width=20, height=20),
        thickness=0.5,
    )

    gen = TPMSGenerator(
        config,
        target_geometry_path=dummy_target,
    )
    mesh, metadata = gen.generate_mesh(allow_nonwatertight=True)
    written = gen.export(mesh, output_path)

    assert isinstance(mesh, trimesh.Trimesh)
    assert isinstance(metadata, MeshQualityMetadata)
    assert not mesh.is_empty
    assert metadata.triangle_count > 0
    assert isinstance(metadata.warnings, list)
    assert np.isfinite(np.asarray(metadata.bbox)).all()
    assert written == output_path
    assert output_path.exists()


def test_grading_spec_radial(tmp_path):
    config = _base_config(
        tpms_type=TPMSType.GYROID,
        voxels_per_cell=20,
        domain=DomainConfig(length=20, width=20, height=20),
        thickness=GradingSpec(
            kind="radial",
            params={
                "t_center": 0.2,
                "t_outer": 0.8,
                "center": (10, 10, 10),
                "radius": 10.0,
            },
        ),
    )

    gen = TPMSGenerator(config)
    mesh, metadata = gen.generate_mesh(allow_nonwatertight=True)

    assert isinstance(mesh, trimesh.Trimesh)
    assert isinstance(metadata, MeshQualityMetadata)
    assert not mesh.is_empty
    assert metadata.triangle_count > 0
    assert isinstance(metadata.warnings, list)
    assert np.isfinite(np.asarray(metadata.bbox)).all()


def test_target_mesh_argument():
    config = _base_config(
        tpms_type=TPMSType.GYROID,
        voxels_per_cell=20,
        domain=DomainConfig(length=20, width=20, height=20),
    )
    target = trimesh.primitives.Box(extents=(10, 10, 10))

    gen = TPMSGenerator(config, target_mesh=target)
    mesh, metadata = gen.generate_mesh(allow_nonwatertight=True)

    assert isinstance(mesh, trimesh.Trimesh)
    assert isinstance(metadata, MeshQualityMetadata)
    assert not mesh.is_empty


def test_target_geometry_path_and_target_mesh_are_mutually_exclusive(tmp_path):
    config = _base_config(
        tpms_type=TPMSType.GYROID,
        voxels_per_cell=20,
        domain=DomainConfig(length=20, width=20, height=20),
    )
    dummy_target = tmp_path / "dummy_target.stl"
    trimesh.primitives.Box(extents=(10, 10, 10)).export(dummy_target)

    with pytest.raises(ValueError, match="either target_geometry_path or target_mesh"):
        TPMSGenerator(
            config,
            target_geometry_path=dummy_target,
            target_mesh=trimesh.primitives.Box(extents=(8, 8, 8)),
        )


def test_invalid_thickness_type_rejected():
    with pytest.raises(ValidationError):
        GenerationConfig(geometry={"thickness": "invalid"})


def test_default_thickness_is_constant():
    config = GenerationConfig()
    gen = TPMSGenerator(config)
    assert gen.grading_spec.kind == "constant"
    assert gen.grading_spec.params["t"] == 0.5


def test_grading_spec_constant_requires_t():
    with pytest.raises(ValidationError):
        GradingSpec(kind="constant", params={})


def test_grading_spec_rejects_unknown_affine_params():
    with pytest.raises(ValidationError):
        GradingSpec(kind="affine", params={"bx": 0.1, "foo": 1.0})


def test_grading_spec_affine_rejects_invalid_bounds():
    with pytest.raises(ValidationError):
        GradingSpec(kind="affine", params={"tmin": 1.0, "tmax": 0.5})


def test_grading_spec_radial_requires_positive_radius():
    with pytest.raises(ValidationError):
        GradingSpec(
            kind="radial",
            params={"t_center": 0.2, "t_outer": 0.8, "center": (0, 0, 0), "radius": 0.0},
        )


def test_grading_spec_radial_center_must_be_3d():
    with pytest.raises(ValidationError):
        GradingSpec(
            kind="radial",
            params={"t_center": 0.2, "t_outer": 0.8, "center": (0, 0), "radius": 1.0},
        )


def test_lids_accepts_lid_spec():
    config = GenerationConfig(geometry=GeometryConfig(lids=LidSpec(z_min=2.0)))
    assert config.geometry.lids.z_min == 2.0
    assert config.geometry.lids.x_min == 0.0


def test_lids_rejects_unknown_key():
    with pytest.raises(ValidationError):
        GenerationConfig(geometry={"lids": {"foo": 1.0}})


def test_lids_rejects_negative_thickness():
    with pytest.raises(ValidationError):
        GenerationConfig(geometry={"lids": {"x_min": -0.1}})


def test_sampling_rejects_negative_margin():
    with pytest.raises(ValidationError):
        GenerationConfig(sampling={"margin_cells": -0.1})


def test_booleans_reject_negative_lid_overlap_margin():
    with pytest.raises(ValidationError):
        GenerationConfig(booleans={"lid_overlap_margin": -0.1})


def _generator_for_quality_tests() -> TPMSGenerator:
    config = _base_config(
        tpms_type=TPMSType.GYROID,
        voxels_per_cell=20,
        domain=DomainConfig(length=20, width=20, height=20),
        thickness=0.5,
    )
    return TPMSGenerator(config)


def test_generate_mesh_can_skip_watertight_check():
    gen = _generator_for_quality_tests()
    mesh, metadata = gen.generate_mesh(allow_nonwatertight=True, check_watertight=False)

    assert isinstance(mesh, trimesh.Trimesh)
    assert metadata.is_watertight is None
    assert isinstance(metadata.warnings, list)


def test_quality_gate_rejects_empty_mesh():
    gen = _generator_for_quality_tests()
    empty_mesh = trimesh.Trimesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.int64))

    with pytest.raises(ValueError, match="empty"):
        gen._validate_mesh_quality(
            empty_mesh,
            allow_nonwatertight=False,
            check_watertight=True,
        )


def test_quality_gate_rejects_non_finite_vertices():
    gen = _generator_for_quality_tests()
    mesh = trimesh.creation.box()
    mesh.vertices[0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        gen._validate_mesh_quality(
            mesh,
            allow_nonwatertight=False,
            check_watertight=True,
        )


def test_quality_gate_rejects_non_watertight_by_default():
    gen = _generator_for_quality_tests()
    mesh = trimesh.creation.box()
    mesh.update_faces(np.arange(len(mesh.faces) - 1))
    mesh.remove_unreferenced_vertices()
    assert not mesh.is_watertight

    with pytest.raises(ValueError, match="not watertight"):
        gen._validate_mesh_quality(
            mesh,
            allow_nonwatertight=False,
            check_watertight=True,
        )


def test_quality_gate_allows_non_watertight_with_warning():
    gen = _generator_for_quality_tests()
    mesh = trimesh.creation.box()
    mesh.update_faces(np.arange(len(mesh.faces) - 1))
    mesh.remove_unreferenced_vertices()
    assert not mesh.is_watertight

    metadata = gen._validate_mesh_quality(
        mesh,
        allow_nonwatertight=True,
        check_watertight=True,
    )
    assert metadata.is_watertight is False
    assert any("not watertight" in warning for warning in metadata.warnings)


def test_repeat_runs_are_deterministic_for_same_config_and_target():
    config = _base_config(
        tpms_type=TPMSType.GYROID,
        cell_size=10.0,
        voxels_per_cell=18,
        domain=DomainConfig(length=24, width=24, height=24),
    )
    target = trimesh.primitives.Box(extents=(18, 18, 18))

    mesh_a, _ = TPMSGenerator(config, target_mesh=target).generate_mesh(allow_nonwatertight=True)
    mesh_b, _ = TPMSGenerator(config, target_mesh=target).generate_mesh(allow_nonwatertight=True)

    assert _mesh_signature(mesh_a) == _mesh_signature(mesh_b)


def test_margin_cells_changes_grid_shape():
    config_a = _base_config(margin_cells=0.5)
    config_b = _base_config(margin_cells=1.0)

    gen_a = TPMSGenerator(config_a)
    gen_b = TPMSGenerator(config_b)
    gen_a.generate_grid()
    gen_b.generate_grid()

    assert gen_a.x.shape != gen_b.x.shape


def test_center_target_mesh_toggle_affects_target_centroid():
    target = trimesh.primitives.Box(extents=(10, 10, 10))
    target.apply_translation([4.0, -3.0, 2.0])

    centered = _base_config(booleans=BooleanConfig(center_target_mesh=True))
    not_centered = _base_config(booleans=BooleanConfig(center_target_mesh=False))

    centered_mesh = TPMSGenerator(centered, target_mesh=target).get_target_mesh()
    not_centered_mesh = TPMSGenerator(not_centered, target_mesh=target).get_target_mesh()

    assert np.allclose(centered_mesh.bounding_box.centroid, np.zeros(3), atol=1e-8)
    assert not np.allclose(not_centered_mesh.bounding_box.centroid, np.zeros(3), atol=1e-8)


def test_clip_target_to_domain_toggle_affects_output_bounds():
    domain = DomainConfig(length=20, width=20, height=20)
    large_target = trimesh.primitives.Box(extents=(200, 200, 200))

    clipped = _base_config(
        domain=domain,
        booleans=BooleanConfig(clip_target_to_domain=True),
    )
    unclipped = _base_config(
        domain=domain,
        booleans=BooleanConfig(clip_target_to_domain=False),
    )

    clipped_mesh, _ = TPMSGenerator(clipped, target_mesh=large_target).generate_mesh(allow_nonwatertight=True)
    unclipped_mesh, _ = TPMSGenerator(unclipped, target_mesh=large_target).generate_mesh(allow_nonwatertight=True)

    clipped_extent_x = clipped_mesh.bounds[1][0] - clipped_mesh.bounds[0][0]
    unclipped_extent_x = unclipped_mesh.bounds[1][0] - unclipped_mesh.bounds[0][0]

    assert clipped_extent_x <= domain.length + 1e-6
    assert unclipped_extent_x > domain.length
