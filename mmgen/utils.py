"""Mesh utility helpers for boolean operations and exports."""

import os
import trimesh
import numpy as np

_MANIFOLD_INSTALL_MSG = (
    "manifold3d is required for boolean mesh operations. "
    "Install it with `pip install manifold3d` (or add it to your environment dependencies)."
)


def _load_manifold():
    """Import manifold3d types used by boolean operations.

    Returns
    -------
    tuple[type, type]
        ``(Manifold, Mesh)`` classes from ``manifold3d``.

    Raises
    ------
    ImportError
        If ``manifold3d`` is not installed.
    """
    try:
        from manifold3d import Manifold, Mesh
    except ImportError as exc:
        raise ImportError(_MANIFOLD_INSTALL_MSG) from exc
    return Manifold, Mesh


def _to_manifold(mesh: trimesh.Trimesh):
    """Convert a trimesh mesh into a manifold3d manifold object.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Source mesh.

    Returns
    -------
    Any
        manifold3d manifold instance.
    """
    Manifold, Mesh = _load_manifold()
    verts = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.uint32)
    return Manifold(Mesh(vert_properties=verts, tri_verts=faces))


def _to_trimesh(mesh):
    """Convert a manifold3d mesh payload back to ``trimesh.Trimesh``.

    Parameters
    ----------
    mesh : Any
        manifold3d mesh payload exposing ``vert_properties`` and ``tri_verts``.

    Returns
    -------
    trimesh.Trimesh
        Converted mesh with ``process=False``.
    """
    return trimesh.Trimesh(vertices=mesh.vert_properties, faces=mesh.tri_verts, process=False)


def mesh_intersection(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> trimesh.Trimesh:
    """Perform boolean intersection using manifold3d.

    Parameters
    ----------
    mesh_a : trimesh.Trimesh
        First input mesh.
    mesh_b : trimesh.Trimesh
        Second input mesh.

    Returns
    -------
    trimesh.Trimesh
        Intersection mesh.
    """
    m_a = _to_manifold(mesh_a)
    m_b = _to_manifold(mesh_b)
    return _to_trimesh((m_a ^ m_b).to_mesh())

def mesh_union(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> trimesh.Trimesh:
    """Perform boolean union using manifold3d.

    Parameters
    ----------
    mesh_a : trimesh.Trimesh
        First input mesh.
    mesh_b : trimesh.Trimesh
        Second input mesh.

    Returns
    -------
    trimesh.Trimesh
        Union mesh.
    """
    m_a = _to_manifold(mesh_a)
    m_b = _to_manifold(mesh_b)
    return _to_trimesh((m_a + m_b).to_mesh())

def export_mesh(mesh: trimesh.Trimesh, path: str):
    """Export a mesh to disk.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to export.
    path : str
        Output file path. ``.off`` is handled explicitly; other formats are delegated
        to ``trimesh``.
    """
    file_ext = os.path.splitext(path)[1].lower()
    
    if file_ext == '.off':
        off_data = mesh.export(file_type="off")
        if isinstance(off_data, (bytes, bytearray, memoryview)):
            with open(path, "wb") as f:
                f.write(bytes(off_data))
        elif isinstance(off_data, str):
            with open(path, "w", encoding="utf-8") as f:
                f.write(off_data)
        else:
            raise TypeError(
                f"Unsupported OFF export payload type: {type(off_data).__name__}"
            )
    else:
        # Generic export for STL, 3MF, etc.
        mesh.export(path)

def ensure_dir(path: str):
    """Ensure the parent directory for a path exists.

    Parameters
    ----------
    path : str
        File path whose parent directory should be created.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
