import os
from pathlib import Path

from mmgen.config import DomainConfig, GeneratorConfig, TPMSParams
from mmgen.generator import TPMSGenerator
from mmgen.grading import GradingSpec
from mmgen.tpms_types import TPMSType


def test_basic_gyroid():
    """Generates a simple Gyroid block with constant thickness."""
    config = GeneratorConfig(
        tpms=TPMSParams(type=TPMSType.GYROID, cell_size=10.0, resolution=30),
        domain=DomainConfig(length=30, width=30, height=30),
    )
    gen = TPMSGenerator(config, thickness=0.5)
    mesh, metadata = gen.generate_mesh(allow_nonwatertight=True)
    print(f"Metadata: {metadata}")
    gen.export(mesh, "basic_gyroid.stl")


def test_graded_schwarz_p():
    """Generates a graded Schwarz P block."""
    config = GeneratorConfig(
        tpms=TPMSParams(type="schwarz_p", cell_size=10.0, resolution=20),
        domain=DomainConfig(length=30, width=20, height=20),
        lids={"x_min": 2.0, "x_max": 2.0},
    )

    grading_spec = GradingSpec(
        kind="affine",
        params={"a": 0.2, "bx": (0.8 - 0.2) / 50.0, "by": 0.0, "bz": 0.0},
    )

    gen = TPMSGenerator(config, thickness=grading_spec)
    mesh, metadata = gen.generate_mesh(allow_nonwatertight=True)
    print(f"Metadata: {metadata}")
    gen.export(mesh, "graded_schwarz_p.stl")


def test_lids_with_benchy():
    """Generates a Lidinoid pattern with lids at the bottom and top, intersected with Benchy."""
    benchy_path = "3DBenchy.stl"

    config = GeneratorConfig(
        tpms=TPMSParams(type=TPMSType.LIDINOID, cell_size=10.0, resolution=30),
        domain=DomainConfig(length=60, width=40, height=48),
        lids={"z_min": 2.0, "z_max": 2.0},
    )
    target_geom = benchy_path if os.path.exists(benchy_path) else None
    gen = TPMSGenerator(
        config,
        thickness=0.5,
        target_geometry_path=Path(target_geom) if target_geom else None,
    )
    mesh, metadata = gen.generate_mesh(allow_nonwatertight=True)
    print(f"Metadata: {metadata}")
    gen.export(mesh, "lidinoid_lids_benchy.stl")


def test_lidinoid_with_benchy():
    """Generates a Lidinoid pattern intersected with a Benchy STL if available."""
    benchy_path = "3DBenchy.stl"

    config = GeneratorConfig(
        tpms=TPMSParams(type=TPMSType.LIDINOID, cell_size=10.0, resolution=30),
        domain=DomainConfig(length=60, width=40, height=48),
    )
    target_geom = benchy_path if os.path.exists(benchy_path) else None
    gen = TPMSGenerator(
        config,
        thickness=0.5,
        target_geometry_path=Path(target_geom) if target_geom else None,
    )
    mesh, metadata = gen.generate_mesh(allow_nonwatertight=True)
    print(f"Metadata: {metadata}")
    gen.export(mesh, "lidinoid_mesh.stl")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("--- Running Basic Gyroid Test ---")
    test_basic_gyroid()

    print("\n--- Running Graded Schwarz P Test ---")
    test_graded_schwarz_p()

    print("\n--- Running Lids with Benchy Test ---")
    test_lids_with_benchy()

    print("\n--- Running Lidinoid Test ---")
    test_lidinoid_with_benchy()

    print("\nVerification complete. Check generated STL files.")
