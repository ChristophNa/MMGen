# from dataclasses import dataclass, field
# from typing import Optional, Union
# import numpy as np
# from .tpms_types import TPMSType

# @dataclass
# class TPMSParams:
#     """Basic parameters for TPMS generation."""
#     type: TPMSType = TPMSType.GYROID
#     cell_size: float = 10.0  # [mm]
#     resolution: int = 30  # Number of voxels per unit cell

# @dataclass
# class DomainConfig:
#     """Physical dimensions of the domain."""
#     length: float = 50.0  # [mm] (x)
#     width: float = 10.0   # [mm] (y)
#     height: float = 10.0  # [mm] (z)

# @dataclass
# class GeneratorConfig:
#     """Combined configuration for the TPMS generator."""
#     tpms: TPMSParams = field(default_factory=TPMSParams)
#     domain: DomainConfig = field(default_factory=DomainConfig)
#     lids: dict[str, float] = field(default_factory=dict)  # {'z_min': 1.0, ...}

# from __future__ import annotations

from typing import Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator

from .tpms_types import TPMSType

_ALLOWED_LID_KEYS = {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}


class TPMSParams(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    type: TPMSType = TPMSType.GYROID
    cell_size: float = Field(default=10.0, gt=0.0)      # [mm]
    resolution: int = Field(default=20, ge=2)           # voxels per unit cell

    @field_validator("type", mode="before")
    @classmethod
    def parse_tpms_type(cls, v: Any) -> Any:
        # Allow: TPMSType.GYROID (already ok), "GYROID", "gyroid"
        if isinstance(v, str):
            key = v.strip().upper()
            try:
                return TPMSType[key]     # by name
            except KeyError as e:
                raise ValueError(f"Unknown TPMSType '{v}'. Allowed: {[m.name for m in TPMSType]}") from e
        return v


class DomainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    length: float = Field(default=10.0, gt=0.0)  # [mm]
    width: float = Field(default=10.0, gt=0.0)   # [mm]
    height: float = Field(default=10.0, gt=0.0)  # [mm]


class GeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    tpms: TPMSParams = Field(default_factory=TPMSParams)
    domain: DomainConfig = Field(default_factory=DomainConfig)
    lids: Dict[str, float] = Field(default_factory=dict)

    @field_validator("lids")
    @classmethod
    def validate_lids(cls, lids: Dict[str, float]) -> Dict[str, float]:
        unknown = set(lids) - _ALLOWED_LID_KEYS
        if unknown:
            raise ValueError(f"Unknown lid keys: {sorted(unknown)}; allowed: {sorted(_ALLOWED_LID_KEYS)}")

        negatives = {k: v for k, v in lids.items() if v < 0}
        if negatives:
            raise ValueError(f"Lid thickness must be >= 0; got: {negatives}")

        return lids
