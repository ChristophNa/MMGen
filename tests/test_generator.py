import pytest
import trimesh
from pydantic import ValidationError

from mmgen.config import GeneratorConfig, DomainConfig, TPMSParams, LidSpec
from mmgen.generator import TPMSGenerator
from mmgen.grading import GradingSpec
from mmgen.tpms_types import TPMSType


def test_basic_gyroid(tmp_path):
    """Generates a simple Gyroid block."""
    output_name = tmp_path / "basic_gyroid"
    output_path = tmp_path / "basic_gyroid.stl"

    config = GeneratorConfig(
        tpms=TPMSParams(type=TPMSType.GYROID, cell_size=10.0, resolution=30),
        domain=DomainConfig(length=30, width=30, height=30),
    )

    gen = TPMSGenerator(config, thickness=0.5, output_name=str(output_name))
    mesh = gen.run()

    assert isinstance(mesh, trimesh.Trimesh)
    assert not mesh.is_empty
    assert output_path.exists()


def test_graded_schwarz_p(tmp_path):
    """Generates an affine-graded Schwarz P block."""
    output_name = tmp_path / "graded_schwarz_p"
    output_path = tmp_path / "graded_schwarz_p.stl"

    config = GeneratorConfig(
        tpms=TPMSParams(type=TPMSType.SCHWARZ_P, cell_size=10.0, resolution=30),
        domain=DomainConfig(length=50, width=20, height=20),
    )

    grading_spec = GradingSpec(
        kind="affine",
        params={"a": 0.2, "bx": (0.8 - 0.2) / 50.0, "by": 0.0, "bz": 0.0},
    )

    gen = TPMSGenerator(config, thickness=grading_spec, output_name=str(output_name))
    mesh = gen.run()

    assert isinstance(mesh, trimesh.Trimesh)
    assert not mesh.is_empty
    assert output_path.exists()


def test_lidinoid_with_target(tmp_path):
    """Generates a Lidinoid pattern and checks target intersection handling."""
    output_name = tmp_path / "lidinoid_mesh"
    output_path = tmp_path / "lidinoid_mesh.stl"

    dummy_target = tmp_path / "dummy_target.stl"
    box = trimesh.primitives.Box(extents=(10, 10, 10))
    box.export(dummy_target)

    config = GeneratorConfig(
        tpms=TPMSParams(type=TPMSType.LIDINOID, cell_size=5.0, resolution=20),
        domain=DomainConfig(length=20, width=20, height=20),
    )

    gen = TPMSGenerator(
        config,
        thickness=0.5,
        target_geometry=str(dummy_target),
        output_name=str(output_name),
    )
    mesh = gen.run()

    assert isinstance(mesh, trimesh.Trimesh)
    assert not mesh.is_empty
    assert output_path.exists()


def test_grading_spec_radial(tmp_path):
    """Test passing a radial grading spec."""
    output_name = tmp_path / "custom_grading"

    config = GeneratorConfig(
        tpms=TPMSParams(type=TPMSType.GYROID, cell_size=10.0, resolution=20),
        domain=DomainConfig(length=20, width=20, height=20),
    )

    custom_grading = GradingSpec(
        kind="radial",
        params={
            "t_center": 0.2,
            "t_outer": 0.8,
            "center": (10, 10, 10),
            "radius": 10.0,
        },
    )

    gen = TPMSGenerator(config, thickness=custom_grading, output_name=str(output_name))
    mesh = gen.run()

    assert isinstance(mesh, trimesh.Trimesh)
    assert not mesh.is_empty


def test_missing_thickness_error():
    """Ensure TypeError is raised if thickness is not provided."""
    config = GeneratorConfig(
        tpms=TPMSParams(type=TPMSType.GYROID),
        domain=DomainConfig(length=10, width=10, height=10),
    )
    with pytest.raises(TypeError):
        TPMSGenerator(config)


def test_invalid_thickness_type():
    """Ensure ValueError is raised if thickness is invalid type."""
    config = GeneratorConfig()
    with pytest.raises(ValueError):
        TPMSGenerator(config, thickness="invalid")


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
    config = GeneratorConfig(lids=LidSpec(z_min=2.0))
    assert config.lids.z_min == 2.0
    assert config.lids.x_min == 0.0


def test_lids_rejects_unknown_key():
    with pytest.raises(ValidationError):
        GeneratorConfig(lids={"foo": 1.0})


def test_lids_rejects_negative_thickness():
    with pytest.raises(ValidationError):
        GeneratorConfig(lids={"x_min": -0.1})
