import numpy as np
import trimesh
from skimage import measure
from .config import GeneratorConfig, GradingParams
from .tpms_types import TPMS_REGISTRY
from .utils import pycork_intersection

class TPMSGenerator:
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.domain = config.domain
        self.tpms_params = config.tpms
        
        # Adjust grading parameters if xl is not set
        if self.config.grading and self.config.grading.xl is None:
            self.config.grading.xl = self.domain.length
            
    def generate_grid(self):
        """Generates the 3D grid based on domain and resolution."""
        # Expand domain slightly to ensure crisp boundaries during clipping/intersection
        # Adding half a cell size margin on each side
        margin = self.tpms_params.cell_size * 0.5
        
        nx = (self.domain.length + 2 * margin) / self.tpms_params.cell_size
        ny = (self.domain.width + 2 * margin) / self.tpms_params.cell_size
        nz = (self.domain.height + 2 * margin) / self.tpms_params.cell_size
        
        res = self.tpms_params.resolution
        # Sampling slightly outside the requested domain [0, L]
        self.x, self.y, self.z = np.mgrid[
            -margin : self.domain.length + margin : res * nx,
            -margin : self.domain.width + margin : res * ny,
            -margin : self.domain.height + margin : res * nz
        ]
        
    def evaluate_field(self) -> np.ndarray:
        """Evaluates the scalar field based on the selected TPMS and grading."""
        eq_func = TPMS_REGISTRY[self.tpms_params.type]
        
        # Base TPMS value (without level-set 't')
        field = eq_func(self.x, self.y, self.z, self.tpms_params.cell_size)
        
        # Apply Grading
        if self.config.grading:
            g = self.config.grading
            # Linear slope for the 't' parameter: t(x) = t0 + m * (x - x0)
            m = (g.tl - g.t0) / (g.xl - g.x0)
            t_field = g.t0 + m * (self.x - g.x0)
            # In the original graded_tpms.py, the final field is (eq) - (t)**2
            # or for some types, it was slightly different. 
            # We'll stick to the "Double" variant logic where we subtract t**2.
            final_field = field - t_field**2
        else:
            # Default t value if no grading (using t0 as static level-set)
            # If grading is None, we might want a default level-set value in TPMSParams but let's assume 0.5
            t_static = 0.5 
            final_field = field - t_static**2
            
        return final_field

    def apply_boundary_conditions(self, vol: np.ndarray):
        """Ensures vol = 1 at coordinates boundaries to close the mesh (as in graded_tpms.py)."""
        # This is a trick to make marching cubes produce a closed mesh at the box boundaries
        vol[0, :, :] = 1.0
        vol[-1, :, :] = 1.0
        vol[:, 0, :] = 1.0
        vol[:, -1, :] = 1.0
        vol[:, :, 0] = 1.0
        vol[:, :, -1] = 1.0

    def generate_raw_mesh(self) -> trimesh.Trimesh:
        """Runs marching cubes to create the initial TPMS mesh."""
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
        """Loads target geometry or creates a default box."""
        if self.config.target_geometry:
            mesh = trimesh.load_mesh(self.config.target_geometry)
            # Center the target mesh at the origin
            mesh.vertices -= mesh.bounding_box.centroid
            return mesh
        else:
            # Create a box matching the domain, centered at [0,0,0] by default
            return trimesh.primitives.Box(extents=(self.domain.length, self.domain.width, self.domain.height))

    def run(self) -> trimesh.Trimesh:
        """Executes the full generation and intersection process."""
        print(f"Generating TPMS: {self.tpms_params.type.name}...")
        tpms_mesh = self.generate_raw_mesh()
        
        print("Preparing target geometry...")
        target_mesh = self.get_target_mesh()
        
        # Always center the TPMS mesh at the origin for consistent intersection
        tpms_mesh.vertices -= [self.domain.length/2, self.domain.width/2, self.domain.height/2]

        print("Performing PyCork intersection...")
        final_mesh = pycork_intersection(tpms_mesh, target_mesh)
        
        output_path = f"{self.config.output_name}.stl"
        final_mesh.export(output_path)
        print(f"Mesh saved to {output_path}")
        
        return final_mesh
