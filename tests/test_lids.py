import logging
import os
import sys

import numpy as np
import trimesh

# Add parent directory to path to import mmgen
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmgen.config import DomainConfig, GeneratorConfig, TPMSParams
from mmgen.generator import TPMSGenerator
from mmgen.tpms_types import TPMSType

logger = logging.getLogger(__name__)


def check_lid_coverage(
    mesh: trimesh.Trimesh,
    domain: DomainConfig,
    axis: int,
    value: float,
    tolerance: float = 0.5,
):
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
    logger.debug("Mesh bounds: %s", bounds)

    on_plane = np.abs(mesh.vertices[:, axis] - value) < tolerance

    count = np.sum(on_plane)
    logger.info(
        "Checked plane %d=%s (tol=%s). Vertices found: %d",
        axis,
        value,
        tolerance,
        count,
    )

    return count > 0


def main():
    # Test 1: Z_MIN Lid
    logging.basicConfig(
        level=os.getenv("MMGEN_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("--- Test 1: Z_MIN Lid (Bottom) ---")
    config = GeneratorConfig(
        tpms=TPMSParams(type=TPMSType.GYROID, cell_size=10.0, resolution=20),
        domain=DomainConfig(length=20, width=20, height=20),
        lids={"z_min": 2.0},
    )

    gen = TPMSGenerator(config, thickness=0.5)
    mesh, metadata = gen.generate_mesh(allow_nonwatertight=True)
    logger.info("Metadata: %s", metadata)

    has_lid = check_lid_coverage(mesh, config.domain, 2, -10.0)

    if has_lid:
        logger.info("PASS: Lid detected at Z_MIN.")
    else:
        logger.error("FAIL: No Lid detected at Z_MIN.")

    # Test 2: Double Lids (X_MIN, X_MAX)
    logger.info("--- Test 2: X_MIN and X_MAX Lids ---")
    config2 = GeneratorConfig(
        tpms=TPMSParams(type=TPMSType.SCHWARZ_P, cell_size=10.0, resolution=20),
        domain=DomainConfig(length=20, width=20, height=20),
        lids={"x_min": 2.0, "x_max": 2.0},
    )

    gen2 = TPMSGenerator(config2, thickness=0.5)
    mesh2, metadata2 = gen2.generate_mesh(allow_nonwatertight=True)
    logger.info("Metadata: %s", metadata2)

    has_lid_min = check_lid_coverage(mesh2, config2.domain, 0, -10.0)
    has_lid_max = check_lid_coverage(mesh2, config2.domain, 0, 10.0)

    if has_lid_min and has_lid_max:
        logger.info("PASS: Lids detected at X_MIN and X_MAX.")
    else:
        logger.error("FAIL: X_MIN=%s, X_MAX=%s", has_lid_min, has_lid_max)


if __name__ == "__main__":
    main()
