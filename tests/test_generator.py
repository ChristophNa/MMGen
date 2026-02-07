import pytest
import os
import trimesh
from mmgen.config import GeneratorConfig, DomainConfig, TPMSParams, GradingParams
from mmgen.tpms_types import TPMSType
from mmgen.generator import TPMSGenerator

def test_basic_gyroid(tmp_path):
    """Generates a simple Gyroid block."""
    output_name = tmp_path / "basic_gyroid"
    config = GeneratorConfig(
        tpms=TPMSParams(type=TPMSType.GYROID, cell_size=10.0, resolution=30j),
        domain=DomainConfig(length=30, width=30, height=30),
        output_name=str(output_name)
    )
    gen = TPMSGenerator(config)
    # Mocking the run method's export to avoid writing to actual disk or just use tmp_path
    # The run method exports to {output_name}.stl
    
    # We can run the generator
    mesh = gen.run()
    
    # Check if mesh is valid
    assert isinstance(mesh, trimesh.Trimesh)
    assert not mesh.is_empty
    
    # Check if file exists
    expected_file = output_name.with_suffix(".stl")
    assert expected_file.exists()

def test_graded_schwarz_p(tmp_path):
    """Generates a graded Schwarz P block."""
    output_name = tmp_path / "graded_schwarz_p"
    config = GeneratorConfig(
        tpms=TPMSParams(type=TPMSType.SCHWARZ_P, cell_size=10.0, resolution=30j),
        domain=DomainConfig(length=50, width=20, height=20),
        grading=GradingParams(t0=0.2, tl=0.8),
        output_name=str(output_name)
    )
    gen = TPMSGenerator(config)
    mesh = gen.run()
    
    assert isinstance(mesh, trimesh.Trimesh)
    assert not mesh.is_empty
    expected_file = output_name.with_suffix(".stl")
    assert expected_file.exists()

def test_lidinoid_with_benchy(tmp_path):
    """Generates a Lidinoid pattern and checks for intersection handling."""
    # We might not have 3DBenchy.stl in the test environment.
    # We can skip the target_geometry or create a dummy STL.
    
    output_name = tmp_path / "lidinoid_mesh"
    
    # Create a dummy target mesh
    dummy_target = tmp_path / "dummy_target.stl"
    box = trimesh.primitives.Box(extents=(10, 10, 10))
    box.export(dummy_target)
    
    config = GeneratorConfig(
        tpms=TPMSParams(type=TPMSType.LIDINOID, cell_size=5.0, resolution=20j),
        domain=DomainConfig(length=20, width=20, height=20), # Smaller domain for faster test
        target_geometry=str(dummy_target),
        output_name=str(output_name)
    )
    gen = TPMSGenerator(config)
    mesh = gen.run()
    
    assert isinstance(mesh, trimesh.Trimesh)
    assert not mesh.is_empty
    expected_file = output_name.with_suffix(".stl")
    assert expected_file.exists()
