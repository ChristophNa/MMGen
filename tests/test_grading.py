import numpy as np

from mmgen.grading import GradingSpec, grading_from_spec


def test_grading_from_spec_constant_returns_constant_vector():
    spec = GradingSpec(kind="constant", params={"t": 0.4})
    points = np.array([[0.0, 0.0, 0.0], [1.0, -2.0, 3.0]])

    values = grading_from_spec(spec)(points)

    assert values.shape == (2,)
    assert np.allclose(values, np.array([0.4, 0.4]))


def test_grading_from_spec_affine_applies_coefficients_and_clamping():
    spec = GradingSpec(
        kind="affine",
        params={"a": 0.1, "bx": 0.5, "by": -0.25, "bz": 0.0, "tmin": 0.0, "tmax": 1.0},
    )
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, -2.0, 0.0],
            [-4.0, 4.0, 0.0],
        ]
    )

    values = grading_from_spec(spec)(points)

    expected = np.array([0.1, 0.6, 1.0, 0.0])
    assert np.allclose(values, expected)


def test_grading_from_spec_radial_interpolates_and_saturates_at_radius():
    spec = GradingSpec(
        kind="radial",
        params={"center": (0.0, 0.0, 0.0), "radius": 2.0, "t_center": 0.2, "t_outer": 0.8},
    )
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )

    values = grading_from_spec(spec)(points)

    assert values[0] == 0.2
    assert values[1] == 0.5
    assert values[2] == 0.8
    assert values[3] == 0.8
