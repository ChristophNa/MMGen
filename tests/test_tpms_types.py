import numpy as np

from mmgen.tpms_types import TPMS_REGISTRY, TPMSType


def test_registry_contains_all_enum_members():
    assert set(TPMS_REGISTRY) == set(TPMSType)


def test_all_tpms_functions_return_finite_grid_values():
    x, y, z = np.mgrid[0:1:3j, 0:1:3j, 0:1:3j]

    for tpms_type, eq_func in TPMS_REGISTRY.items():
        values = eq_func(x, y, z, 10.0)
        assert values.shape == x.shape, f"Unexpected shape for {tpms_type.name}"
        assert np.isfinite(values).all(), f"Non-finite values for {tpms_type.name}"
