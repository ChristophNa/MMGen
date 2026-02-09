import argparse
import logging
import os
from pathlib import Path

from mmgen.config import (
    DomainConfig,
    GenerationConfig,
    GeometryConfig,
    LidSpec,
    LatticeConfig,
    SamplingConfig,
)
from mmgen.generator import TPMSGenerator
from mmgen.grading import GradingSpec
from mmgen.tpms_types import TPMSType

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_console_logging(level: str = "INFO") -> None:
    """Configure root logging to write to stderr.

    Parameters
    ----------
    level : str, optional
        Logging level name.
    """
    logging.basicConfig(
        level=level.upper(),
        format=LOG_FORMAT,
    )


def configure_file_logging(log_path: str = "mmgen.log") -> None:
    """Configure root logging to write to a file.

    Parameters
    ----------
    log_path : str, optional
        Output log file path.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        filename=log_path,
        filemode="w",
    )


def configure_warning_only_logging() -> None:
    """Configure root logging at ``WARNING`` level."""
    logging.basicConfig(
        level=logging.WARNING,
        format=LOG_FORMAT,
    )


def test_basic_gyroid():
    """Generate a gyroid example with constant thickness.

    Notes
    -----
    This is a sample workflow that demonstrates passing a dedicated logger.
    """
    config = GenerationConfig(
        lattice=LatticeConfig(type=TPMSType.GYROID, cell_size=10.0),
        sampling=SamplingConfig(voxels_per_cell=30),
        geometry=GeometryConfig(
            domain=DomainConfig(length=30, width=30, height=30),
            thickness=0.5,
        ),
    )
    task_logger = logging.getLogger("mmgen.examples.basic_gyroid")
    gen = TPMSGenerator(config, logger=task_logger)
    mesh, metadata = gen.generate_mesh(allow_nonwatertight=True)
    logger.info("Metadata: %s", metadata)
    gen.export(mesh, Path("basic_gyroid.stl"))

def test_basic_lidinoid():
    """Generate a lidinoid example with lids and affine grading.

    Notes
    -----
    This is a sample workflow that demonstrates passing a dedicated logger.
    """
    config = GenerationConfig(
        lattice=LatticeConfig(type=TPMSType.LIDINOID, cell_size=10.0),
        sampling=SamplingConfig(voxels_per_cell=40),
        geometry=GeometryConfig(
            domain=DomainConfig(length=30, width=20, height=20),
            thickness=0.6,
            lids=LidSpec(x_min=2.0, x_max=2.0),
        ),
    )
    grading_spec = GradingSpec(
        kind="affine",
        params={"a": 0.5, "bx": (1.0 - 0.5) / 30.0, "by": 0.0, "bz": 0.0},
    )
    config.geometry.thickness = grading_spec
    task_logger = logging.getLogger("mmgen.examples.basic_lidinoid")
    gen = TPMSGenerator(config, logger=task_logger)
    mesh, metadata = gen.generate_mesh(allow_nonwatertight=True)
    logger.info("Metadata: %s", metadata)
    gen.export(mesh, Path("basic_lidinoid.stl"))


def test_graded_schwarz_p():
    """Generate a Schwarz-P example with affine thickness grading."""
    config = GenerationConfig(
        lattice=LatticeConfig(type=TPMSType.SCHWARZ_P, cell_size=10.0),
        sampling=SamplingConfig(voxels_per_cell=30),
        geometry=GeometryConfig(
            domain=DomainConfig(length=30, width=20, height=20),
            lids=LidSpec(x_min=2.0, x_max=2.0),
        ),
    )

    grading_spec = GradingSpec(
        kind="affine",
        params={"a": 0.2, "bx": (0.8 - 0.2) / 30.0, "by": 0.0, "bz": 0.0},
    )

    config.geometry.thickness = grading_spec
    gen = TPMSGenerator(config)
    mesh, metadata = gen.generate_mesh(allow_nonwatertight=True)
    logger.info("Metadata: %s", metadata)
    gen.export(mesh, Path("graded_schwarz_p.stl"))


def test_lids_with_benchy():
    """Generate lidinoid with top/bottom lids, clipped by Benchy when available."""
    benchy_path = "3DBenchy.stl"

    config = GenerationConfig(
        lattice=LatticeConfig(type=TPMSType.LIDINOID, cell_size=10.0),
        sampling=SamplingConfig(voxels_per_cell=30),
        geometry=GeometryConfig(
            domain=DomainConfig(length=60, width=40, height=48),
            lids=LidSpec(z_min=2.0, z_max=2.0),
            thickness=0.5,
        ),
    )
    target_geom = benchy_path if os.path.exists(benchy_path) else None
    gen = TPMSGenerator(
        config,
        target_geometry_path=Path(target_geom) if target_geom else None,
    )
    mesh, metadata = gen.generate_mesh(allow_nonwatertight=True)
    logger.info("Metadata: %s", metadata)
    gen.export(mesh, Path("lidinoid_lids_benchy.stl"))


def test_lidinoid_with_benchy():
    """Generate lidinoid clipped by Benchy when available."""
    benchy_path = "3DBenchy.stl"

    config = GenerationConfig(
        lattice=LatticeConfig(type=TPMSType.LIDINOID, cell_size=10.0),
        sampling=SamplingConfig(voxels_per_cell=30),
        geometry=GeometryConfig(
            domain=DomainConfig(length=60, width=40, height=48),
            thickness=0.5,
        ),
    )
    target_geom = benchy_path if os.path.exists(benchy_path) else None
    gen = TPMSGenerator(
        config,
        target_geometry_path=Path(target_geom) if target_geom else None,
    )
    mesh, metadata = gen.generate_mesh(allow_nonwatertight=True)
    logger.info("Metadata: %s", metadata)
    gen.export(mesh, Path("lidinoid_mesh.stl"))


def parse_args() -> argparse.Namespace:
    """Parse CLI options for sample generation workflows.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.

    Notes
    -----
    CLI options:

    ``--log-mode``
        One of ``console``, ``file``, or ``warning``.

    ``--log-level``
        Logging level string used when ``--log-mode=console``.

    ``--log-file``
        Output path used when ``--log-mode=file``.
    """
    parser = argparse.ArgumentParser(description="Run MMGen sample generation workflows.")
    parser.add_argument(
        "--log-mode",
        choices=("console", "file", "warning"),
        default="console",
        help="Logging output mode: console, file, or warning-only console.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Console log level (used only when --log-mode=console).",
    )
    parser.add_argument(
        "--log-file",
        default="mmgen.log",
        help="Log file path (used only when --log-mode=file).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Usage examples:
    #   python main.py --log-mode console --log-level INFO
    #   python main.py --log-mode file --log-file mmgen.log
    #   python main.py --log-mode warning
    args = parse_args()
    if args.log_mode == "file":
        configure_file_logging(args.log_file)
    elif args.log_mode == "warning":
        configure_warning_only_logging()
    else:
        configure_console_logging(args.log_level)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    logger.info("--- Running Basic Gyroid Test ---")
    test_basic_gyroid()

    logger.info("--- Running Basic Lidinoid Test ---")
    test_basic_lidinoid()

    logger.info("--- Running Graded Schwarz P Test ---")
    test_graded_schwarz_p()

    logger.info("--- Running Lids with Benchy Test ---")
    test_lids_with_benchy()

    logger.info("--- Running Lidinoid Test ---")
    test_lidinoid_with_benchy()

    logger.info("Verification complete. Check generated STL files.")
