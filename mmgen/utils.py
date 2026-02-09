import os
import trimesh
import numpy as np

_MANIFOLD_INSTALL_MSG = (
    "manifold3d is required for boolean mesh operations. "
    "Install it with `pip install manifold3d` (or add it to your environment dependencies)."
)


def _load_manifold():
    try:
        from manifold3d import Manifold, Mesh
    except ImportError as exc:
        raise ImportError(_MANIFOLD_INSTALL_MSG) from exc
    return Manifold, Mesh


def _to_manifold(mesh: trimesh.Trimesh):
    Manifold, Mesh = _load_manifold()
    verts = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.uint32)
    return Manifold(Mesh(vert_properties=verts, tri_verts=faces))


def _to_trimesh(mesh):
    return trimesh.Trimesh(vertices=mesh.vert_properties, faces=mesh.tri_verts, process=False)


def mesh_intersection(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Perform a boolean intersection using manifold3d.
    """
    m_a = _to_manifold(mesh_a)
    m_b = _to_manifold(mesh_b)
    return _to_trimesh((m_a ^ m_b).to_mesh())

def mesh_union(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Perform a boolean union using manifold3d.
    """
    m_a = _to_manifold(mesh_a)
    m_b = _to_manifold(mesh_b)
    return _to_trimesh((m_a + m_b).to_mesh())

def export_mesh(mesh: trimesh.Trimesh, path: str):
    """
    Export a mesh to a file. Supports .off, .stl, .3mf, etc. via trimesh.
    """
    file_ext = os.path.splitext(path)[1].lower()
    
    if file_ext == '.off':
        # Specific handling for OFF to match previous behavior if needed, 
        # but trimesh.export handles it too. Keeping previous logic for safety.
        off_data = trimesh.exchange.off.export_off(mesh, digits=6)
        with open(path, "w") as f:
            f.write(off_data)
    else:
        # Generic export for STL, 3MF, etc.
        mesh.export(path)

def ensure_dir(path: str):
    """Ensure directory exists."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
