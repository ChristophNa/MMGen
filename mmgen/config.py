from dataclasses import dataclass, field
from typing import Optional, Union
import numpy as np
from .tpms_types import TPMSType

@dataclass
class TPMSParams:
    """Basic parameters for TPMS generation."""
    type: TPMSType = TPMSType.GYROID
    cell_size: float = 10.0  # [mm]
    resolution: complex = 30j  # Number of voxels per unit cell

@dataclass
class GradingParams:
    """Parameters for linear grading of the 't' parameter."""
    t0: float = 0.2
    tl: float = 0.8
    x0: float = 0.0
    xl: Optional[float] = None  # If None, will be set to domain length

@dataclass
class DomainConfig:
    """Physical dimensions of the domain."""
    length: float = 50.0  # [mm] (x)
    width: float = 10.0   # [mm] (y)
    height: float = 10.0  # [mm] (z)

@dataclass
class GeneratorConfig:
    """Combined configuration for the TPMS generator."""
    tpms: TPMSParams = field(default_factory=TPMSParams)
    domain: DomainConfig = field(default_factory=DomainConfig)
    grading: Optional[GradingParams] = None
    target_geometry: Optional[str] = None  # Path to STL file
    output_name: str = "result"
