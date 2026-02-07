import numpy as np
from typing import Callable, Union

# Define GradingFunc type alias
GradingFunc = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

def constant(value: float) -> GradingFunc:
    """Returns a constant thickness function."""
    def func(x, y, z):
        return np.full_like(x, value, dtype=float)
    return func

def linear_x(t0: float, tl: float, x0: float, xl: float) -> GradingFunc:
    """Returns a linear gradient function along the X axis."""
    def func(x, y, z):
        # t(x) = t0 + m * (x - x0)
        m = (tl - t0) / (xl - x0)
        return t0 + m * (x - x0)
    return func

def linear_y(t0: float, tl: float, y0: float, yl: float) -> GradingFunc:
    """Returns a linear gradient function along the Y axis."""
    def func(x, y, z):
        # t(y) = t0 + m * (y - y0)
        m = (tl - t0) / (yl - y0)
        return t0 + m * (y - y0)
    return func


def linear_z(t0: float, tl: float, z0: float, zl: float) -> GradingFunc:
    """Returns a linear gradient function along the Z axis."""
    def func(x, y, z):
        # t(z) = t0 + m * (z - z0)
        m = (tl - t0) / (zl - z0)
        return t0 + m * (z - z0)
    return func

def radial(t_center: float, t_outer: float, center: tuple[float, float, float], radius: float) -> GradingFunc:
    """Returns a radial gradient function from a center point."""
    def func(x, y, z):
        # Calculate distance from center
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
        # scale dist so that at radius it is 1.0 (or just linear interpolation based on radius)
        # t(d) = t_center + m * dist
        # at dist=0, t=t_center
        # at dist=radius, t=t_outer
        m = (t_outer - t_center) / radius
        return t_center + m * dist
    return func
