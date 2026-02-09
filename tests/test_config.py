import pytest
from pydantic import ValidationError

from mmgen.config import LatticeConfig, LidSpec
from mmgen.tpms_types import TPMSType


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("gyroid", TPMSType.GYROID),
        ("SCHWARZ_P", TPMSType.SCHWARZ_P),
        ("lidinoid", TPMSType.LIDINOID),
    ],
)
def test_parse_tpms_type_is_case_insensitive(value: str, expected: TPMSType):
    config = LatticeConfig(type=value)
    assert config.type == expected


def test_parse_tpms_type_rejects_unknown_value():
    with pytest.raises(ValidationError, match="Unknown TPMSType"):
        LatticeConfig(type="unknown_surface")


def test_lidspec_enabled_returns_only_positive_entries():
    lids = LidSpec(x_min=1.0, x_max=0.0, y_max=0.5, z_min=0.0)
    assert lids.enabled() == {"x_min": 1.0, "y_max": 0.5}
