import pytest
import os
import trimesh
import numpy as np
from mmgen.config import GeneratorConfig, DomainConfig, TPMSParams
from mmgen.tpms_types import TPMSType
from mmgen.generator import TPMSGenerator
from mmgen import grading

def test_basic_gyroid(tmp_path):
    """Generates a simple Gyroid block."""
    output_name = tmp_path / "basic_gyroid"
    config = GeneratorConfig(
        tpms=TPMSParams(type=TPMSType.GYROID, cell_size=10.0, resolution=30j),
        domain=DomainConfig(length=30, width=30, height=30),
        output_name=str(output_name)
    )
    # Pass constant thickness
    gen = TPMSGenerator(config, thickness=0.5)
    
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
        output_name=str(output_name)
    )
    
    # Linear grading
    grading_func = grading.linear_x(t0=0.2, tl=0.8, x0=0.0, xl=50.0)
    
    gen = TPMSGenerator(config, thickness=grading_func)
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
    # constant thickness
    gen = TPMSGenerator(config, thickness=0.5)
    mesh = gen.run()
    
    assert isinstance(mesh, trimesh.Trimesh)
    assert not mesh.is_empty
    expected_file = output_name.with_suffix(".stl")
    assert expected_file.exists()

def test_custom_grading_function(tmp_path):
    """Test passing a custom lambda as grading function."""
    output_name = tmp_path / "custom_grading"
    config = GeneratorConfig(
        tpms=TPMSParams(type=TPMSType.GYROID, cell_size=10.0, resolution=20j),
        domain=DomainConfig(length=20, width=20, height=20),
        output_name=str(output_name)
    )
    
    # Custom radial grading: t = 0.2 at center, 0.8 at radius 10
    # Center is (10, 10, 10) for a 20x20x20 box centered at 0? 
    # Wait, domain logic in generator:
    # box creates grid from 0 to L. So center is (10, 10, 10).
    
    custom_grading = grading.radial(t_center=0.2, t_outer=0.8, center=(10, 10, 10), radius=10.0)
    
    gen = TPMSGenerator(config, thickness=custom_grading)
    mesh = gen.run()
    
    assert isinstance(mesh, trimesh.Trimesh)
    assert not mesh.is_empty

def test_missing_thickness_error(tmp_path):
    """Ensure ValueError is raised if thickness is not provided."""
    config = GeneratorConfig(
        tpms=TPMSParams(type=TPMSType.GYROID),
        domain=DomainConfig(length=10, width=10, height=10)
    )
    with pytest.raises(TypeError): # TypeError because argument is missing
        TPMSGenerator(config)

def test_invalid_thickness_type(tmp_path):
    """Ensure ValueError is raised if thickness is invalid type."""
    config = GeneratorConfig()
    with pytest.raises(ValueError):
        TPMSGenerator(config, thickness="invalid")
