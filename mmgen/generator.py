"""Core TPMS mesh generation pipeline."""

from dataclasses import dataclass, field
import logging
from pathlib import Path

import numpy as np
import trimesh
from skimage import measure

from .config import GenerationConfig
from .grading import GradingSpec, grading_from_spec
from .tpms_types import TPMS_REGISTRY
from .utils import mesh_intersection, mesh_union


@dataclass
class MeshQualityMetadata:
    """Quality metadata captured for a generated mesh.

    Attributes
    ----------
    triangle_count : int
        Number of mesh faces.
    bbox : tuple[tuple[float, float, float], tuple[float, float, float]]
        Axis-aligned bounding box as ``(min_xyz, max_xyz)``.
    is_watertight : bool or None
        Watertightness result when checked, otherwise ``None``.
    warnings : list[str]
        Non-fatal quality warnings gathered during validation.
    """

    triangle_count: int
    bbox: tuple[tuple[float, float, float], tuple[float, float, float]]
    is_watertight: bool | None
    warnings: list[str] = field(default_factory=list)

class TPMSGenerator:
    """Generate TPMS meshes from validated configuration models.

    Parameters
    ----------
    config : GenerationConfig
        Full generation configuration.
    target_geometry_path : Path or None, optional
        Optional path to a target geometry file used for final intersection.
    target_mesh : trimesh.Trimesh or None, optional
        Optional in-memory target mesh used for final intersection.
    logger : logging.Logger or None, optional
        Logger instance used for diagnostics.
    log_level : int or str or None, optional
        Optional logger level override.

    Raises
    ------
    ValueError
        If both ``target_geometry_path`` and ``target_mesh`` are provided.

    Notes
    -----
    High-level workflow in :meth:`generate_mesh`:
    1. Evaluate TPMS field on a voxel grid.
    2. Extract an isosurface with marching cubes.
    3. Optionally union face lids.
    4. Build or load target geometry.
    5. Optionally clip target geometry to the configured domain.
    6. Intersect TPMS and target meshes.
    7. Validate mesh quality and return metadata.
    """

    def __init__(
        self,
        config: GenerationConfig,
        target_geometry_path: Path | None = None,
        target_mesh: trimesh.Trimesh | None = None,
        logger: logging.Logger | None = None,
        log_level: int | str | None = None,
    ):
        """Initialize a TPMS generator instance.

        Parameters
        ----------
        config : GenerationConfig
            Full generation configuration.
        target_geometry_path : Path or None, optional
            Optional path to a target geometry file used for final intersection.
        target_mesh : trimesh.Trimesh or None, optional
            Optional in-memory target mesh used for final intersection.
        logger : logging.Logger or None, optional
            Logger instance used for diagnostics.
        log_level : int or str or None, optional
            Optional logger level override.
        """
        self.config = config
        self.domain = config.geometry.domain
        self.lattice = config.lattice
        self.sampling = config.sampling
        self.boolean_config = config.booleans
        self.target_geometry_path = Path(target_geometry_path) if target_geometry_path is not None else None
        self.target_mesh = target_mesh
        self.logger = logger or logging.getLogger(__name__)
        if log_level is not None:
            self.logger.setLevel(self._normalize_log_level(log_level))

        if self.target_geometry_path is not None and self.target_mesh is not None:
            raise ValueError("Provide either target_geometry_path or target_mesh, not both.")

        thickness = config.geometry.thickness
        if isinstance(thickness, (float, int)):
            self.grading_spec = GradingSpec(kind="constant", params={"t": float(thickness)})
        else:
            self.grading_spec = thickness

        self.grading_func = grading_from_spec(self.grading_spec)

    @staticmethod
    def _normalize_log_level(log_level: int | str) -> int:
        """Normalize integer or named logging level to its integer value.

        Parameters
        ----------
        log_level : int or str
            Logging level as numeric value or name (for example ``"INFO"``).

        Returns
        -------
        int
            Integer logging level.

        Raises
        ------
        ValueError
            If the level name is unknown.
        """
        if isinstance(log_level, int):
            return log_level
        if isinstance(log_level, str):
            level = logging.getLevelNamesMapping().get(log_level.upper())
            if level is not None:
                return level
        raise ValueError(f"Invalid log level: {log_level!r}")
            
    def generate_grid(self):
        """Generate sampling grids spanning the domain plus configured margin.

        Notes
        -----
        The generated arrays are stored on the instance as ``self.x``, ``self.y``,
        and ``self.z``.
        """
        # Expand domain slightly to ensure crisp boundaries during clipping/intersection
        margin = self.lattice.cell_size * self.sampling.margin_cells
        
        nx = (self.domain.length + 2 * margin) / self.lattice.cell_size
        ny = (self.domain.width + 2 * margin) / self.lattice.cell_size
        nz = (self.domain.height + 2 * margin) / self.lattice.cell_size
        
        res = complex(0, self.sampling.voxels_per_cell)
        # Sampling slightly outside the requested domain [0, L]
        self.x, self.y, self.z = np.mgrid[
            -margin : self.domain.length + margin : res * nx,
            -margin : self.domain.width + margin : res * ny,
            -margin : self.domain.height + margin : res * nz
        ]
        
    def evaluate_field(self) -> np.ndarray:
        """Evaluate the graded TPMS scalar field on the current sampling grid.

        Returns
        -------
        ndarray
            Scalar field values with the same shape as ``self.x``.
        """
        eq_func = TPMS_REGISTRY[self.lattice.type]
        
        # Base TPMS value (without level-set 't')
        field = eq_func(self.x, self.y, self.z, self.lattice.cell_size)
        
        # Apply Grading / Thickness
        # Evaluate grading over flattened XYZ points, then reshape to grid.
        points = np.column_stack((self.x.ravel(), self.y.ravel(), self.z.ravel()))
        t_field = self.grading_func(points).reshape(self.x.shape)
        
        # We'll stick to the "Double" variant logic where we subtract t**2.
        final_field = field - t_field**2
            
        return final_field

    def apply_boundary_conditions(self, vol: np.ndarray):
        """Force boundary voxels positive to encourage closed marching-cubes output.

        Parameters
        ----------
        vol : ndarray
            Scalar field volume modified in place.
        """
        # This is a trick to make marching cubes produce a closed mesh at the box boundaries
        vol[0, :, :] = 1.0
        vol[-1, :, :] = 1.0
        vol[:, 0, :] = 1.0
        vol[:, -1, :] = 1.0
        vol[:, :, 0] = 1.0
        vol[:, :, -1] = 1.0

    def generate_raw_mesh(self) -> trimesh.Trimesh:
        """Run marching cubes to generate an unclipped TPMS mesh.

        Returns
        -------
        trimesh.Trimesh
            Raw TPMS mesh before target clipping and lid boolean operations.
        """
        self.generate_grid()
        vol = self.evaluate_field()
        self.apply_boundary_conditions(vol)
        
        # Calculate spacing for marching cubes
        # spacing should reflect the physical size covered by the grid
        grid_lx = self.x[-1,0,0] - self.x[0,0,0]
        grid_ly = self.y[0,-1,0] - self.y[0,0,0]
        grid_lz = self.z[0,0,-1] - self.z[0,0,0]

        spacing = (
            grid_lx / (vol.shape[0] - 1),
            grid_ly / (vol.shape[1] - 1),
            grid_lz / (vol.shape[2] - 1)
        )
        
        verts, faces, normals, values = measure.marching_cubes(vol, 0, spacing=spacing)
        
        # Shift vertices so (0,0,0) in grid aligns with (0,0,0) in world space
        verts += [self.x[0,0,0], self.y[0,0,0], self.z[0,0,0]]
        
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)
        return mesh

    def get_target_mesh(self) -> trimesh.Trimesh:
        """Load or create the mesh used as final clipping target.

        Returns
        -------
        trimesh.Trimesh
            Target mesh in centered coordinates when configured.

        Raises
        ------
        ValueError
            If a loaded scene contains no geometry.

        Notes
        -----
        Source selection order:
        1. ``target_mesh`` argument (copied).
        2. ``target_geometry_path`` (loaded from disk).
        3. Fallback domain-aligned box primitive.
        """
        if self.target_mesh is not None:
            self.logger.info("Using provided target mesh.")
            mesh = self.target_mesh.copy()
        elif self.target_geometry_path is not None:
            self.logger.info("Loading target geometry from: %s", self.target_geometry_path)
            mesh = trimesh.load_mesh(self.target_geometry_path)
            self.logger.debug("Loaded mesh type: %s", type(mesh))
        else:
            # Create a box matching the domain, centered at [0,0,0] by default
            return trimesh.primitives.Box(extents=(self.domain.length, self.domain.width, self.domain.height))

        # Handle Scene object if returned
        if isinstance(mesh, trimesh.Scene):
            self.logger.info("Target is a Scene; concatenating geometry into a single mesh.")
            if len(mesh.geometry) == 0:
                 raise ValueError("Loaded Scene is empty!")
            # Concatenate all geometries in the scene
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

        # Normalize to a mutable Trimesh (primitives can be immutable).
        mesh = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices, dtype=float).copy(),
            faces=np.asarray(mesh.faces).copy(),
            process=False,
        )

        self.logger.debug("Target mesh vertices shape: %s", mesh.vertices.shape)
        self.logger.debug("Target mesh bounds: %s", mesh.bounds)

        # Center the target mesh at the origin
        # mesh.bounding_box might trigger generation, bounds is property
        if mesh.bounds is None:
            self.logger.warning("Target mesh bounds are None.")

        if self.boolean_config.center_target_mesh:
            mesh.vertices -= mesh.bounding_box.centroid
        return mesh

    def _generate_lid(self, side: str, thickness: float) -> trimesh.Trimesh | None:
        """Create a rectangular lid mesh aligned to one domain face.

        Parameters
        ----------
        side : str
            Face identifier (for example ``"x_min"`` or ``"z_max"``).
        thickness : float
            Lid thickness in millimeters.

        Returns
        -------
        trimesh.Trimesh or None
            Generated lid mesh, or ``None`` for unknown ``side``.

        Notes
        -----
        Supported ``side`` values are ``x_min``, ``x_max``, ``y_min``,
        ``y_max``, ``z_min``, and ``z_max``.
        """
        l, w, h = self.domain.length, self.domain.width, self.domain.height
        
        # Margin to ensure overlap
        margin = self.boolean_config.lid_overlap_margin 
        
        # Lids are generated relative to the Origin (0,0,0) which is the center of the domain.
        # Domain extends [-L/2, L/2], [-W/2, W/2], [-H/2, H/2]
        
        if side == 'x_min':
            # Lid at x = -L/2. Thickness extends inwards? 
            # Or outwards? Ideally inwards to be valid part of domain.
            # If Lid is part of the structure, it occupies [-L/2, -L/2 + t].
            center_x = -l/2 + thickness/2
            box = trimesh.primitives.Box(extents=(thickness, w + margin, h + margin))
            box.apply_translation([center_x, 0, 0])
            
        elif side == 'x_max':
            # Lid at x = L/2. Occupies [L/2 - t, L/2].
            center_x = l/2 - thickness/2
            box = trimesh.primitives.Box(extents=(thickness, w + margin, h + margin))
            box.apply_translation([center_x, 0, 0])
            
        elif side == 'y_min':
            center_y = -w/2 + thickness/2
            box = trimesh.primitives.Box(extents=(l + margin, thickness, h + margin))
            box.apply_translation([0, center_y, 0])
            
        elif side == 'y_max':
            center_y = w/2 - thickness/2
            box = trimesh.primitives.Box(extents=(l + margin, thickness, h + margin))
            box.apply_translation([0, center_y, 0])
            
        elif side == 'z_min':
            center_z = -h/2 + thickness/2
            box = trimesh.primitives.Box(extents=(l + margin, w + margin, thickness))
            box.apply_translation([0, 0, center_z])
            
        elif side == 'z_max':
            center_z = h/2 - thickness/2
            box = trimesh.primitives.Box(extents=(l + margin, w + margin, thickness))
            box.apply_translation([0, 0, center_z])
        else:
            self.logger.warning("Unknown lid side '%s'; skipping.", side)
            return None
            
        return box

    @staticmethod
    def _compute_metadata(
        mesh: trimesh.Trimesh, check_watertight: bool, warnings: list[str]
    ) -> MeshQualityMetadata:
        """Build metadata summary for a generated mesh.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            Mesh to summarize.
        check_watertight : bool
            Whether ``mesh.is_watertight`` should be queried.
        warnings : list[str]
            Accumulated non-fatal warnings.

        Returns
        -------
        MeshQualityMetadata
            Mesh quality metadata.
        """
        bounds = np.asarray(mesh.bounds, dtype=float)
        bbox = (
            tuple(float(v) for v in bounds[0]),
            tuple(float(v) for v in bounds[1]),
        )
        is_watertight = bool(mesh.is_watertight) if check_watertight else None
        return MeshQualityMetadata(
            triangle_count=int(len(mesh.faces)),
            bbox=bbox,
            is_watertight=is_watertight,
            warnings=warnings,
        )

    def _validate_mesh_quality(
        self,
        mesh: trimesh.Trimesh,
        *,
        allow_nonwatertight: bool,
        check_watertight: bool,
    ) -> MeshQualityMetadata:
        """Validate core mesh quality constraints and return metadata.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            Mesh to validate.
        allow_nonwatertight : bool
            Whether non-watertight meshes are allowed with warning metadata.
        check_watertight : bool
            Whether watertightness should be evaluated.

        Returns
        -------
        MeshQualityMetadata
            Validation metadata and warnings.

        Raises
        ------
        ValueError
            If the mesh is empty, contains non-finite vertices, or fails
            watertightness requirements.
        """
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ValueError("Generated mesh is empty: expected non-empty vertices and faces.")

        vertices = np.asarray(mesh.vertices)
        if not np.isfinite(vertices).all():
            raise ValueError("Generated mesh contains non-finite vertex coordinates.")

        warnings: list[str] = []
        if check_watertight and not mesh.is_watertight:
            warnings.append("Generated mesh is not watertight.")
            self.logger.warning("Generated mesh is not watertight.")
            if not allow_nonwatertight:
                raise ValueError(
                    "Generated mesh is not watertight. "
                    "Pass allow_nonwatertight=True to continue with warning metadata."
                )

        return self._compute_metadata(mesh, check_watertight=check_watertight, warnings=warnings)

    def generate_mesh(
        self,
        *,
        allow_nonwatertight: bool = False,
        check_watertight: bool = True,
    ) -> tuple[trimesh.Trimesh, MeshQualityMetadata]:
        """Generate, combine, clip, and validate a TPMS mesh.

        Parameters
        ----------
        allow_nonwatertight : bool, optional
            Allow non-watertight output with warning metadata.
        check_watertight : bool, optional
            Evaluate watertightness during quality validation.

        Returns
        -------
        tuple[trimesh.Trimesh, MeshQualityMetadata]
            Final mesh and associated quality metadata.

        Notes
        -----
        Main behavior toggles:

        ``allow_nonwatertight``
            If ``True``, non-watertight meshes are returned with warning metadata.

        ``check_watertight``
            If ``False``, watertightness is not queried and metadata contains
            ``is_watertight=None``.
        """
        self.logger.info("Generating TPMS: %s", self.lattice.type.name)
        tpms_mesh = self.generate_raw_mesh()

        # Center TPMS Mesh First (0..L -> -L/2..L/2)
        tpms_mesh.vertices -= [self.domain.length / 2, self.domain.width / 2, self.domain.height / 2]

        # Apply Lids (Generated in Centered Coordinates)
        enabled_lids = self.config.geometry.lids.enabled()
        if enabled_lids:
            self.logger.info("Generating and unioning %d lids.", len(enabled_lids))
            for side, thickness in enabled_lids.items():
                lid_mesh = self._generate_lid(side, thickness)
                if lid_mesh:
                    # mesh_union handles the union using manifold3d.
                    self.logger.debug("Adding lid: %s (thickness=%s)", side, thickness)
                    tpms_mesh = mesh_union(tpms_mesh, lid_mesh)

        self.logger.info("Preparing target geometry.")
        target_mesh = self.get_target_mesh()

        # Clip target mesh to domain dimensions to avoid TPMS margins affecting the result
        if self.boolean_config.clip_target_to_domain:
            self.logger.info("Clipping target geometry to domain boundaries.")
            domain_box = trimesh.primitives.Box(
                extents=(self.domain.length, self.domain.width, self.domain.height)
            )
            # domain_box is centered at 0,0,0, matching our coordinate system
            target_mesh = mesh_intersection(target_mesh, domain_box)

        self.logger.info("Performing mesh intersection (TPMS + lids with target).")
        final_mesh = mesh_intersection(tpms_mesh, target_mesh)
        metadata = self._validate_mesh_quality(
            final_mesh,
            allow_nonwatertight=allow_nonwatertight,
            check_watertight=check_watertight,
        )
        self.logger.info(
            "Mesh generation complete: %d triangles, watertight=%s",
            metadata.triangle_count,
            metadata.is_watertight,
        )

        return final_mesh, metadata

    def export(self, mesh: trimesh.Trimesh, output_path: Path) -> Path:
        """Export a mesh and return the output path.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            Mesh to write.
        output_path : Path
            Destination file path.

        Returns
        -------
        Path
            Path written to disk.
        """
        path = Path(output_path)
        mesh.export(path)
        self.logger.info("Mesh saved to %s", path)
        return path
