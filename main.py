from mmgen.config import GeneratorConfig, DomainConfig, TPMSParams, GradingParams
from mmgen.tpms_types import TPMSType
from mmgen.generator import TPMSGenerator

def test_basic_gyroid():
    """Generates a simple Gyroid block."""
    config = GeneratorConfig(
        tpms=TPMSParams(type=TPMSType.GYROID, cell_size=10.0, resolution=30j),
        domain=DomainConfig(length=30, width=30, height=30),
        output_name="basic_gyroid"
    )
    gen = TPMSGenerator(config)
    gen.run()

def test_graded_schwarz_p():
    """Generates a graded Schwarz P block."""
    config = GeneratorConfig(
        tpms=TPMSParams(type=TPMSType.SCHWARZ_P, cell_size=10.0, resolution=30j),
        domain=DomainConfig(length=50, width=20, height=20),
        grading=GradingParams(t0=0.2, tl=0.8),
        output_name="graded_schwarz_p"
    )
    gen = TPMSGenerator(config)
    gen.run()

def test_lidinoid_with_benchy():
    """Generates a Lidinoid pattern intersected with a Benchy STL if available."""
    # Assuming 3DBenchy.stl is in the same directory as original scripts
    benchy_path = "3DBenchy.stl"
    
    config = GeneratorConfig(
        tpms=TPMSParams(type=TPMSType.LIDINOID, cell_size=5.0, resolution=50j),
        domain=DomainConfig(length=60, width=40, height=40),
        target_geometry=benchy_path if os.path.exists(benchy_path) else None,
        output_name="lidinoid_mesh"
    )
    gen = TPMSGenerator(config)
    gen.run()

if __name__ == "__main__":
    import os
    # Ensure we are in the source directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("--- Running Basic Gyroid Test ---")
    test_basic_gyroid()
    
    print("\n--- Running Graded Schwarz P Test ---")
    test_graded_schwarz_p()
    
    print("\n--- Running Lidinoid Test ---")
    test_lidinoid_with_benchy()
    
    print("\nVerification complete. Check generated STL files.")
