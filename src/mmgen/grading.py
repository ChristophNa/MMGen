"""Thickness grading specifications and evaluators."""

from typing import Callable, Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator


# Input: points with shape (N, 3) -> Output: thickness values with shape (N,)
GradingFunc = Callable[[np.ndarray], np.ndarray]


class GradingSpec(BaseModel):
    """Validated description of a thickness grading law.

    Attributes
    ----------
    kind : {"constant", "affine", "radial"}
        Grading model family.
    params : dict[str, float]
        Model-specific parameters validated according to ``kind``.

    Notes
    -----
    Parameter options by ``kind``:

    ``constant``
        Required: ``t``.

    ``affine``
        Optional: ``a``, ``bx``, ``by``, ``bz``, ``tmin``, ``tmax``.
        Formula: ``t(x, y, z) = a + bx*x + by*y + bz*z`` with optional clamping.

    ``radial``
        Required: ``radius``, ``t_center``, ``t_outer``.
        Optional: ``center_x``, ``center_y``, ``center_z`` (each defaults to ``0.0``).
    """

    kind: Literal["constant", "affine", "radial"]
    params: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_params(self) -> "GradingSpec":
        """Validate and normalize grading parameters in-place.

        Returns
        -------
        GradingSpec
            The current instance with normalized numeric parameters.

        Raises
        ------
        ValueError
            If required parameters are missing, malformed, or inconsistent.
        """
        k = self.kind
        p = dict(self.params)

        if k == "constant":
            allowed = {"t"}
            self._reject_unknown_params(p, allowed, k)
            if "t" not in p:
                raise ValueError("constant grading requires params['t']")
            self.params = p
            return self

        if k == "affine":
            allowed = {"a", "bx", "by", "bz", "tmin", "tmax"}
            self._reject_unknown_params(p, allowed, k)

            if "tmin" in p and "tmax" in p and p["tmin"] > p["tmax"]:
                raise ValueError("affine grading requires tmin <= tmax")

            self.params = p
            return self

        if k == "radial":
            allowed = {"center_x", "center_y", "center_z", "radius", "t_center", "t_outer"}
            self._reject_unknown_params(p, allowed, k)

            for key in ("radius", "t_center", "t_outer"):
                if key not in p:
                    raise ValueError(f"radial grading requires params['{key}']")

            if p["radius"] <= 0.0:
                raise ValueError("radial grading requires radius > 0")

            p["center_x"] = p.get("center_x", 0.0)
            p["center_y"] = p.get("center_y", 0.0)
            p["center_z"] = p.get("center_z", 0.0)

            self.params = p
            return self

        return self

    @staticmethod
    def _reject_unknown_params(params: dict[str, float], allowed: set[str], kind: str) -> None:
        """Raise when unknown keys are found for the selected grading kind.

        Parameters
        ----------
        params : dict[str, float]
            User-provided parameter dictionary.
        allowed : set[str]
            Allowed keys for the grading kind.
        kind : str
            Grading kind name used for the error message.

        Raises
        ------
        ValueError
            If ``params`` contains keys that are not in ``allowed``.
        """
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(
                f"Unknown params for kind '{kind}': {sorted(unknown)}; allowed: {sorted(allowed)}"
            )

def grading_from_spec(spec: GradingSpec) -> GradingFunc:
    """Build a vectorized thickness function from a grading specification.

    Parameters
    ----------
    spec : GradingSpec
        Validated grading specification.

    Returns
    -------
    GradingFunc
        Callable mapping ``(N, 3)`` points to ``(N,)`` thickness values.

    Raises
    ------
    ValueError
        If the grading kind is unknown.

    Notes
    -----
    Implemented grading functions:

    ``constant``
        Returns a constant vector of ``t``.

    ``affine``
        Computes ``a + bx*x + by*y + bz*z`` and optionally clamps to
        ``[tmin, tmax]``.

    ``radial``
        Interpolates from ``t_center`` to ``t_outer`` based on normalized
        distance from ``(center_x, center_y, center_z)``, saturated at ``radius``.
    """
    k = spec.kind
    p = spec.params

    if k == "constant":
        t = float(p["t"])
        return lambda points: np.full((len(points),), t, dtype=float)

    if k == "affine":
        # t(x,y,z) = a + bx*x + by*y + bz*z
        a = float(p.get("a", 0.0))
        bx = float(p.get("bx", 0.0))
        by = float(p.get("by", 0.0))
        bz = float(p.get("bz", 0.0))
        tmin = p.get("tmin", None)
        tmax = p.get("tmax", None)

        def f(points: np.ndarray) -> np.ndarray:
            pts = np.asarray(points, dtype=float)
            t = a + bx * pts[:, 0] + by * pts[:, 1] + bz * pts[:, 2]
            if tmin is not None:
                t = np.maximum(t, float(tmin))
            if tmax is not None:
                t = np.minimum(t, float(tmax))
            return t

        return f

    if k == "radial":
        # t = t_center + (t_outer - t_center) * clamp(r/R, 0..1)
        center = np.array(
            [
                float(p.get("center_x", 0.0)),
                float(p.get("center_y", 0.0)),
                float(p.get("center_z", 0.0)),
            ],
            dtype=float,
        )
        radius = float(p["radius"])
        t_center = float(p["t_center"])
        t_outer = float(p["t_outer"])

        def f(points: np.ndarray) -> np.ndarray:
            pts = np.asarray(points, dtype=float)
            r = np.linalg.norm(pts - center[None, :], axis=1)
            u = np.clip(r / radius, 0.0, 1.0)
            return t_center + (t_outer - t_center) * u

        return f

    raise ValueError(f"Unknown grading kind: {k}")
