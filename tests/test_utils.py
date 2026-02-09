import builtins

import pytest
import trimesh

from mmgen.utils import mesh_intersection, mesh_union


def _box(extents):
    return trimesh.primitives.Box(extents=extents)


@pytest.fixture
def no_manifold_import(monkeypatch):
    original_import = builtins.__import__

    def patched_import(name, *args, **kwargs):
        if name == "manifold3d":
            raise ImportError("No module named 'manifold3d'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", patched_import)


def test_mesh_intersection_requires_manifold3d(no_manifold_import):
    mesh_a = _box((1, 1, 1))
    mesh_b = _box((1, 1, 1))

    with pytest.raises(ImportError, match="Install it with `pip install manifold3d`"):
        mesh_intersection(mesh_a, mesh_b)


def test_mesh_union_requires_manifold3d(no_manifold_import):
    mesh_a = _box((1, 1, 1))
    mesh_b = _box((1, 1, 1))

    with pytest.raises(ImportError, match="Install it with `pip install manifold3d`"):
        mesh_union(mesh_a, mesh_b)
