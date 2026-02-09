import argparse
import logging
import os
from pathlib import Path

from mmgen.config import DomainConfig, GeneratorConfig, TPMSParams
from mmgen.generator import TPMSGenerator
from mmgen.grading import GradingSpec
from mmgen.tpms_types import TPMSType

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_console_logging(level: str = "INFO") -> None:
    """Example: standard console logging."""
    logging.basicConfig(
        level=level.upper(),
        format=LOG_FORMAT,
    )


def configure_file_logging(log_path: str = "mmgen.log") -> None:
    """Example: write logs to a file."""
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        filename=log_path,
        filemode="w",
    )


def configure_warning_only_logging() -> None:
    """Example: suppress INFO/DEBUG by setting level to WARNING."""
    logging.basicConfig(
        level=logging.WARNING,
        format=LOG_FORMAT,
    )


def test_basic_gyroid():
    """Generates a simple Gyroid block with constant thickness.

    Example of passing an explicit logger to TPMSGenerator.
    """
    config = GeneratorConfig(
        tpms=TPMSParams(type=TPMSType.GYROID, cell_size=10.0, resolution=30),
        domain=DomainConfig(length=30, width=30, height=30),
    )
    task_logger = logging.getLogger("mmgen.examples.basic_gyroid")
    gen = TPMSGenerator(config, thickness=0.5, logger=task_logger)
    mesh, metadata = gen.generate_mesh(allow_nonwatertight=True)
    logger.info("Metadata: %s", metadata)
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
    logger.info("Metadata: %s", metadata)
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
    logger.info("Metadata: %s", metadata)
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
    logger.info("Metadata: %s", metadata)
    gen.export(mesh, "lidinoid_mesh.stl")


def parse_args() -> argparse.Namespace:
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

    logger.info("--- Running Graded Schwarz P Test ---")
    test_graded_schwarz_p()

    logger.info("--- Running Lids with Benchy Test ---")
    test_lids_with_benchy()

    logger.info("--- Running Lidinoid Test ---")
    test_lidinoid_with_benchy()

    logger.info("Verification complete. Check generated STL files.")
