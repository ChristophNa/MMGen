from mmgen.config import GeneratorConfig, DomainConfig, TPMSParams
from mmgen.tpms_types import TPMSType
from mmgen.generator import TPMSGenerator
from mmgen import grading

def test_basic_gyroid():
    """Generates a simple Gyroid block with constant thickness."""
    config = GeneratorConfig(
        tpms=TPMSParams(type=TPMSType.GYROID, cell_size=10.0, resolution=30j),
        domain=DomainConfig(length=30, width=30, height=30),
        output_name="basic_gyroid"
    )
    # Pass constant thickness
    gen = TPMSGenerator(config, thickness=0.5)
    gen.run()

def test_graded_schwarz_p():
    """Generates a graded Schwarz P block."""
    config = GeneratorConfig(
        tpms=TPMSParams(type=TPMSType.SCHWARZ_P, cell_size=10.0, resolution=30j),
        domain=DomainConfig(length=30, width=20, height=20),
        output_name="graded_schwarz_p",
        lids={'x_min': 2.0, 'x_max': 2.0} 
    )
    
    # Define linear grading
    # t moves from 0.2 to 0.8 along x-axis from 0 to 50
    grading_func = grading.linear_x(t0=0.2, tl=0.8, x0=0.0, xl=50.0)
    
    gen = TPMSGenerator(config, thickness=grading_func)
    gen.run()

def test_lids_with_benchy():
    """Generates a Lidinoid pattern with lids at the bottom and top, intersected with Benchy."""
    # This demonstrates the 'solid lid' feature on a complex geometry
    benchy_path = "3DBenchy.stl"
    
    config = GeneratorConfig(
        tpms=TPMSParams(type=TPMSType.LIDINOID, cell_size=10.0, resolution=30j),
        domain=DomainConfig(length=60, width=40, height=10),
        target_geometry=benchy_path if os.path.exists(benchy_path) else None,
        output_name="lidinoid_lids_benchy",
        lids={'z_min': 2.0, 'z_max': 2.0} # 2mm solid bottom and top
    )
    # default constant thickness
    gen = TPMSGenerator(config, thickness=0.5)
    gen.run()


def test_lidinoid_with_benchy():
    """Generates a Lidinoid pattern intersected with a Benchy STL if available."""
    # Assuming 3DBenchy.stl is in the same directory as original scripts
    benchy_path = "3DBenchy.stl"
    
    config = GeneratorConfig(
        tpms=TPMSParams(type=TPMSType.LIDINOID, cell_size=10.0, resolution=30j),
        domain=DomainConfig(length=60, width=40, height=10),
        target_geometry=benchy_path if os.path.exists(benchy_path) else None,
        output_name="lidinoid_mesh"
    )
    # default constant thickness
    gen = TPMSGenerator(config, thickness=0.5)
    gen.run()

if __name__ == "__main__":
    import os
    # Ensure we are in the source directory
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

