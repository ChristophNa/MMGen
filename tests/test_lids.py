
import trimesh
import numpy as np
import os
import sys

# Add parent directory to path to import mmgen
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmgen.config import GeneratorConfig, TPMSParams, DomainConfig
from mmgen.generator import TPMSGenerator
from mmgen.tpms_types import TPMSType

def check_lid_coverage(mesh: trimesh.Trimesh, domain: DomainConfig, axis: int, value: float, tolerance: float = 0.5):
    """
    Checks the coverage of the mesh on a specific plane (lid).
    axis: 0 for x, 1 for y, 2 for z
    value: coordinate value of the plane
    tolerance: distance to consider vertices on the plane
    """
    # Find vertices on the plane.
    # Note: Mesh is centered at origin.
    # Domain extents: [-L/2, L/2] etc.
    
    bounds = mesh.bounds
    print(f"Mesh Bounds: {bounds}")
    
    # Check if we have vertices near the requested plane value
    on_plane = np.abs(mesh.vertices[:, axis] - value) < tolerance
    
    count = np.sum(on_plane)
    print(f"Checked plane {axis}={value} (tol={tolerance}). Vertices found: {count}")
    
    # For a solid lid, we expect a distribution of vertices.
    # A simple check: do we have vertices?
    
    return count > 0

def main():
    # Test 1: Z_MIN Lid
    print("--- Test 1: Z_MIN Lid (Bottom) ---")
    config = GeneratorConfig(
        tpms=TPMSParams(type=TPMSType.GYROID, cell_size=10.0, resolution=20),
        domain=DomainConfig(length=20, width=20, height=20),
        lids={'z_min': 2.0}
    )
    
    gen = TPMSGenerator(config, thickness=0.5, output_path="test_lid_z_min.stl")
    mesh = gen.run()
    
    # Check Z=-10 (since height=20, centered)
    # The lid is thickness 2.0, so it goes from -10 to -8.
    # We should see vertices at -10 (bottom face).
    has_lid = check_lid_coverage(mesh, config.domain, 2, -10.0)
    
    if has_lid:
        print("PASS: Lid detected at Z_MIN.")
    else:
        print("FAIL: No Lid detected at Z_MIN.")

    # Test 2: Double Lids (X_MIN, X_MAX)
    print("\n--- Test 2: X_MIN and X_MAX Lids ---")
    config2 = GeneratorConfig(
        tpms=TPMSParams(type=TPMSType.SCHWARZ_P, cell_size=10.0, resolution=20),
        domain=DomainConfig(length=20, width=20, height=20),
        lids={'x_min': 2.0, 'x_max': 2.0}
    )
    
    gen2 = TPMSGenerator(config2, thickness=0.5, output_path="test_lid_x_min_max.stl")
    mesh2 = gen2.run()
    
    # Check X=-10 and X=10
    has_lid_min = check_lid_coverage(mesh2, config2.domain, 0, -10.0)
    has_lid_max = check_lid_coverage(mesh2, config2.domain, 0, 10.0)
    
    if has_lid_min and has_lid_max:
        print("PASS: Lids detected at X_MIN and X_MAX.")
    else:
        print(f"FAIL: X_MIN={has_lid_min}, X_MAX={has_lid_max}")

if __name__ == "__main__":
    main()
