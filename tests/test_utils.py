import builtins

import numpy as np
import pytest
import trimesh

from mmgen.utils import ensure_dir, export_mesh, mesh_intersection, mesh_union


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


def _patch_fake_manifold(monkeypatch):
    captured: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {"inputs": []}

    class FakeMesh:
        def __init__(self, vert_properties, tri_verts):
            captured["inputs"].append((vert_properties, tri_verts))
            self.vert_properties = vert_properties
            self.tri_verts = tri_verts

    class FakeManifoldResult:
        def __init__(self, mesh):
            self._mesh = mesh

        def to_mesh(self):
            return self._mesh

    class FakeManifold:
        def __init__(self, mesh):
            self._mesh = mesh

        def __xor__(self, other):
            verts = np.concatenate([self._mesh.vert_properties, other._mesh.vert_properties], axis=0)
            faces = np.concatenate([self._mesh.tri_verts, other._mesh.tri_verts], axis=0)
            return FakeManifoldResult(FakeMesh(verts, faces))

        def __add__(self, other):
            verts = np.concatenate([self._mesh.vert_properties, other._mesh.vert_properties], axis=0)
            faces = np.concatenate([self._mesh.tri_verts, other._mesh.tri_verts], axis=0)
            return FakeManifoldResult(FakeMesh(verts, faces))

    monkeypatch.setattr("mmgen.utils._load_manifold", lambda: (FakeManifold, FakeMesh))
    return captured


def test_mesh_intersection_uses_fake_manifold_and_returns_trimesh(monkeypatch):
    captured = _patch_fake_manifold(monkeypatch)
    mesh_a = _box((1, 1, 1))
    mesh_b = _box((2, 2, 2))

    out = mesh_intersection(mesh_a, mesh_b)

    assert isinstance(out, trimesh.Trimesh)
    assert len(captured["inputs"]) >= 2
    verts_a, faces_a = captured["inputs"][0]
    assert verts_a.dtype == np.float32
    assert faces_a.dtype == np.uint32


def test_mesh_union_uses_fake_manifold_and_returns_trimesh(monkeypatch):
    captured = _patch_fake_manifold(monkeypatch)
    mesh_a = _box((1, 1, 1))
    mesh_b = _box((2, 2, 2))

    out = mesh_union(mesh_a, mesh_b)

    assert isinstance(out, trimesh.Trimesh)
    assert len(out.vertices) > 0
    assert len(out.faces) > 0
    assert len(captured["inputs"]) >= 2


def test_export_mesh_handles_off_and_stl(tmp_path):
    mesh = trimesh.creation.box()
    off_path = tmp_path / "mesh.off"
    stl_path = tmp_path / "mesh.stl"

    export_mesh(mesh, str(off_path))
    export_mesh(mesh, str(stl_path))

    assert off_path.exists()
    assert stl_path.exists()
    assert off_path.stat().st_size > 0
    assert stl_path.stat().st_size > 0


def test_ensure_dir_creates_nested_directories(tmp_path):
    file_path = tmp_path / "nested" / "path" / "file.stl"

    ensure_dir(str(file_path))

    assert (tmp_path / "nested" / "path").exists()
