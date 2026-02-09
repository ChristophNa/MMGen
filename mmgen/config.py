from typing import Any
from pydantic import BaseModel, Field, ConfigDict, field_validator

from .grading import GradingSpec
from .tpms_types import TPMSType


class LatticeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    type: TPMSType = TPMSType.GYROID
    cell_size: float = Field(default=10.0, gt=0.0)      # [mm]

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


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    voxels_per_cell: int = Field(default=20, ge=2)
    margin_cells: float = Field(default=0.5, ge=0.0)


class DomainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    length: float = Field(default=10.0, gt=0.0)  # [mm]
    width: float = Field(default=10.0, gt=0.0)   # [mm]
    height: float = Field(default=10.0, gt=0.0)  # [mm]


class LidSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    x_min: float = Field(default=0.0, ge=0.0)
    x_max: float = Field(default=0.0, ge=0.0)
    y_min: float = Field(default=0.0, ge=0.0)
    y_max: float = Field(default=0.0, ge=0.0)
    z_min: float = Field(default=0.0, ge=0.0)
    z_max: float = Field(default=0.0, ge=0.0)

    def enabled(self) -> dict[str, float]:
        return {side: thickness for side, thickness in self.model_dump().items() if thickness > 0.0}


class BooleanConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    lid_overlap_margin: float = Field(default=1.0, ge=0.0)
    center_target_mesh: bool = True
    clip_target_to_domain: bool = True


class GeometryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    domain: DomainConfig = Field(default_factory=DomainConfig)
    lids: LidSpec = Field(default_factory=LidSpec)
    thickness: float | GradingSpec = 0.5


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    lattice: LatticeConfig = Field(default_factory=LatticeConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    booleans: BooleanConfig = Field(default_factory=BooleanConfig)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
