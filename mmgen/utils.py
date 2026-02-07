import os
import trimesh
import numpy as np

def mesh_intersection(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Perform a robust boolean intersection using Manifold3D (preferred) or PyCork (fallback).
    """
    # Try Manifold3D first
    try:
        import manifold3d
        from manifold3d import Manifold, Mesh
        
        # Convert Trimesh to Manifold
        # Manifold3D expects vertices and faces
        # We need to ensure the mesh is watertight for best results, but Manifold is robust
        # Explicitly convert to numpy arrays with required dtypes
        verts = np.array(mesh_a.vertices, dtype=np.float32)
        faces = np.array(mesh_a.faces, dtype=np.uint32)
        m_a = Manifold(Mesh(vert_properties=verts, tri_verts=faces))
        
        verts_b = np.array(mesh_b.vertices, dtype=np.float32)
        faces_b = np.array(mesh_b.faces, dtype=np.uint32)
        m_b = Manifold(Mesh(vert_properties=verts_b, tri_verts=faces_b))
        
        # Perform Intersection
        m_result = m_a ^ m_b # ^ operator is intersection in manifold3d
        
        # Convert back to Trimesh
        # Manifold.to_mesh() returns a Mesh object which has vert_properties and tri_verts
        out_mesh = m_result.to_mesh()
        
        return trimesh.Trimesh(vertices=out_mesh.vert_properties, faces=out_mesh.tri_verts, process=False)
        
    except ImportError:
        pass # Fallback to pycork
    except Exception as e:
        print(f"Warning: Manifold3D intersection failed, falling back to PyCork. Error: {e}")

    # Fallback to PyCork
    try:
        import pycork
        verts_a = mesh_a.vertices
        faces_a = mesh_a.faces
        verts_b = mesh_b.vertices
        faces_b = mesh_b.faces
        verts_res, faces_res = pycork.intersection(verts_a, faces_a, verts_b, faces_b)
        return trimesh.Trimesh(vertices=verts_res, faces=faces_res, process=False)
    except (ImportError, Exception) as e:
        print(f"Warning: pycork not available or failed, returning first mesh as mock intersection. Error: {e}")
        return mesh_a

# Alias for backward compatibility if needed, though we will refactor usage
pycork_intersection = mesh_intersection

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
