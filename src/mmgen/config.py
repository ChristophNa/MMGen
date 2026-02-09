"""Configuration models for TPMS mesh generation."""

from typing import Any
from pydantic import BaseModel, Field, ConfigDict, field_validator

from .grading import GradingSpec
from .tpms_types import TPMSType


class LatticeConfig(BaseModel):
    """Lattice selection and unit-cell sizing configuration.

    Attributes
    ----------
    type : TPMSType
        TPMS family used to evaluate the scalar field.
        Supported values are:
        ``TPMSType.GYROID``, ``TPMSType.SCHWARZ_P``, ``TPMSType.DIAMOND``,
        ``TPMSType.LIDINOID``, ``TPMSType.SPLIT_P``, and ``TPMSType.NEOVIUS``.
        String inputs are accepted case-insensitively by enum name
        (for example ``"gyroid"`` or ``"SCHWARZ_P"``).
    cell_size : float
        Unit-cell size in millimeters.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    type: TPMSType = TPMSType.GYROID
    cell_size: float = Field(default=10.0, gt=0.0)      # [mm]

    @field_validator("type", mode="before")
    @classmethod
    def parse_tpms_type(cls, v: Any) -> Any:
        """Parse a TPMS type value from enum or case-insensitive string.

        Parameters
        ----------
        v : Any
            Candidate TPMS type value.

        Returns
        -------
        Any
            Parsed enum value or the original value for downstream validation.

        Raises
        ------
        ValueError
            If a string value does not match any known TPMS type.

        Notes
        -----
        Parsing uses enum *names* (for example ``"SCHWARZ_P"``), not free-form labels.
        """
        # Allow: TPMSType.GYROID (already ok), "GYROID", "gyroid"
        if isinstance(v, str):
            key = v.strip().upper()
            try:
                return TPMSType[key]     # by name
            except KeyError as e:
                raise ValueError(f"Unknown TPMSType '{v}'. Allowed: {[m.name for m in TPMSType]}") from e
        return v


class SamplingConfig(BaseModel):
    """Sampling density and domain margin configuration.

    Attributes
    ----------
    voxels_per_cell : int
        Number of voxel samples along each TPMS cell axis.
        Higher values improve detail at increased memory and runtime cost.
    margin_cells : float
        Additional sampling margin around the domain in cell units.
        This helps robust clipping near domain boundaries.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    voxels_per_cell: int = Field(default=20, ge=2)
    margin_cells: float = Field(default=0.5, ge=0.0)


class DomainConfig(BaseModel):
    """Rectangular domain dimensions in millimeters.

    Attributes
    ----------
    length : float
        Domain size along the x-axis.
    width : float
        Domain size along the y-axis.
    height : float
        Domain size along the z-axis.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    length: float = Field(default=10.0, gt=0.0)  # [mm]
    width: float = Field(default=10.0, gt=0.0)   # [mm]
    height: float = Field(default=10.0, gt=0.0)  # [mm]


class LidSpec(BaseModel):
    """Per-face lid thickness configuration.

    Attributes
    ----------
    x_min : float
        Lid thickness on the minimum x face.
    x_max : float
        Lid thickness on the maximum x face.
    y_min : float
        Lid thickness on the minimum y face.
    y_max : float
        Lid thickness on the maximum y face.
    z_min : float
        Lid thickness on the minimum z face.
    z_max : float
        Lid thickness on the maximum z face.

    Notes
    -----
    Supported side keys are exactly:
    ``x_min``, ``x_max``, ``y_min``, ``y_max``, ``z_min``, ``z_max``.
    A value of ``0.0`` disables a side.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    x_min: float = Field(default=0.0, ge=0.0)
    x_max: float = Field(default=0.0, ge=0.0)
    y_min: float = Field(default=0.0, ge=0.0)
    y_max: float = Field(default=0.0, ge=0.0)
    z_min: float = Field(default=0.0, ge=0.0)
    z_max: float = Field(default=0.0, ge=0.0)

    def enabled(self) -> dict[str, float]:
        """Return only lids with strictly positive thickness.

        Returns
        -------
        dict[str, float]
            Mapping of enabled side names to thickness values.
        """
        return {side: thickness for side, thickness in self.model_dump().items() if thickness > 0.0}


class BooleanConfig(BaseModel):
    """Boolean-operation behavior flags and overlap tuning.

    Attributes
    ----------
    lid_overlap_margin : float
        Extra overlap used when building lids for robust boolean union.
    center_target_mesh : bool
        Whether to center imported target meshes at the origin.
        If ``False``, imported geometry stays in its original coordinates.
    clip_target_to_domain : bool
        Whether to intersect the target mesh with the domain box before final clipping.
        Disable this when you intentionally want target geometry outside the domain
        to influence the final intersection.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    lid_overlap_margin: float = Field(default=1.0, ge=0.0)
    center_target_mesh: bool = True
    clip_target_to_domain: bool = True


class GeometryConfig(BaseModel):
    """Geometry-related generation options.

    Attributes
    ----------
    domain : DomainConfig
        Output domain dimensions.
    lids : LidSpec
        Optional per-face lids.
    thickness : float or GradingSpec
        Constant thickness or position-dependent grading specification.
        Use ``float`` for uniform thickness or ``GradingSpec`` for one of:
        ``constant``, ``affine``, or ``radial`` grading.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    domain: DomainConfig = Field(default_factory=DomainConfig)
    lids: LidSpec = Field(default_factory=LidSpec)
    thickness: float | GradingSpec = 0.5


class GenerationConfig(BaseModel):
    """Top-level validated generation configuration.

    Attributes
    ----------
    lattice : LatticeConfig
        Lattice family and cell settings.
    sampling : SamplingConfig
        Sampling density controls.
    booleans : BooleanConfig
        Boolean operation settings.
    geometry : GeometryConfig
        Domain, lid, and thickness configuration.

    Notes
    -----
    All nested models use ``extra="forbid"`` so unknown keys raise validation errors.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    lattice: LatticeConfig = Field(default_factory=LatticeConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    booleans: BooleanConfig = Field(default_factory=BooleanConfig)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
