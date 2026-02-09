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
    output_path = tmp_path / "basic_gyroid.stl"
    config = GeneratorConfig(
        tpms=TPMSParams(type=TPMSType.GYROID, cell_size=10.0, resolution=30),
        domain=DomainConfig(length=30, width=30, height=30)
    )
    # Pass constant thickness and output_path
    gen = TPMSGenerator(config, thickness=0.5, output_path=str(output_path))
    
    # We can run the generator
    mesh = gen.run()
    
    # Check if mesh is valid
    assert isinstance(mesh, trimesh.Trimesh)
    assert not mesh.is_empty
    
    # Check if file exists
    assert output_path.exists()

def test_graded_schwarz_p(tmp_path):
    """Generates a graded Schwarz P block."""
    output_path = tmp_path / "graded_schwarz_p.stl"
    config = GeneratorConfig(
        tpms=TPMSParams(type=TPMSType.SCHWARZ_P, cell_size=10.0, resolution=30),
        domain=DomainConfig(length=50, width=20, height=20)
    )
    
    # Linear grading
    grading_spec = grading.LinearXGradingSpec(t0=0.2, tl=0.8, x0=0.0, xl=50.0)
    
    gen = TPMSGenerator(config, thickness=grading_spec, output_path=str(output_path))
    mesh = gen.run()
    
    assert isinstance(mesh, trimesh.Trimesh)
    assert not mesh.is_empty
    assert output_path.exists()

def test_lidinoid_with_benchy(tmp_path):
    """Generates a Lidinoid pattern and checks for intersection handling."""
    # We might not have 3DBenchy.stl in the test environment.
    # We can skip the target_geometry or create a dummy STL.
    
    output_path = tmp_path / "lidinoid_mesh.stl"
    
    # Create a dummy target mesh
    dummy_target = tmp_path / "dummy_target.stl"
    box = trimesh.primitives.Box(extents=(10, 10, 10))
    box.export(dummy_target)
    
    config = GeneratorConfig(
        tpms=TPMSParams(type=TPMSType.LIDINOID, cell_size=5.0, resolution=20),
        domain=DomainConfig(length=20, width=20, height=20) # Smaller domain for faster test
    )
    # constant thickness with target_geometry and output_path
    gen = TPMSGenerator(config, thickness=0.5, target_geometry=str(dummy_target), output_path=str(output_path))
    mesh = gen.run()
    
    assert isinstance(mesh, trimesh.Trimesh)
    assert not mesh.is_empty
    assert output_path.exists()

def test_grading_spec_radial(tmp_path):
    """Test passing a radial grading spec."""
    output_path = tmp_path / "custom_grading.stl"
    config = GeneratorConfig(
        tpms=TPMSParams(type=TPMSType.GYROID, cell_size=10.0, resolution=20),
        domain=DomainConfig(length=20, width=20, height=20)
    )
    
    # Custom radial grading: t = 0.2 at center, 0.8 at radius 10
    # Center is (10, 10, 10) for a 20x20x20 box centered at 0? 
    # Wait, domain logic in generator:
    # box creates grid from 0 to L. So center is (10, 10, 10).
    
    custom_grading = grading.RadialGradingSpec(
        t_center=0.2,
        t_outer=0.8,
        center=(10, 10, 10),
        radius=10.0,
    )
    
    gen = TPMSGenerator(config, thickness=custom_grading, output_path=str(output_path))
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
