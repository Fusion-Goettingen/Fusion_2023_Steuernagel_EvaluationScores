"""
Contains function related to the implementation of the Gauss-Wasserstein distance (GWD) and the novel Gaussian
Wasserstein Score (GWS).
"""
import numpy as np
from scipy.linalg import sqrtm
from numpy.linalg import inv

from src.utilities.utils import rot
from src.metrics.abstract_metric import AbstractMetric


class GaussWasserstein(AbstractMetric):
    def __init__(self, squared=True):
        self._squared = squared

    def get_name(self):
        return "GWD"

    def get_label(self):
        return "Squared GWD / $m^2$" if self._squared else "GWD / $m$"

    def calculate(self, m1, p1, m2, p2):
        """
        Calculate gwd for two objects given as
        m: location as [x, y]
        p: shape as [theta, l, w] (orientation theta and semi-axis length l,w)
        :param m1: location as [x, y] of first object
        :param p1: shape as [theta, l, w] (orientation theta and semi-axis length l,w) of first object
        :param m2: location as [x, y] of second object
        :param p2: shape as [theta, l, w] (orientation theta and semi-axis length l,w) of second object
        :return: Squared Gauss Wasserstein distance between both objects
        """
        # numpy-fy and cut down length as necessary
        m1 = np.array(m1)[:2]
        m2 = np.array(m2)[:2]
        p1 = np.array(p1)[:3]
        p2 = np.array(p2)[:3]

        # construct shape matrices
        X1 = rot(p1[0]) @ np.diag(p1[1:] ** 2) @ rot(p1[0]).T
        X2 = rot(p2[0]) @ np.diag(p2[1:] ** 2) @ rot(p2[0]).T

        # prepare calculation
        X1sqrt = rot(p1[0]) @ np.diag(p1[1:]) @ rot(p1[0]).T
        C = sqrtm(X1sqrt @ X2 @ X1sqrt)

        # get distance
        d = np.linalg.norm(m1 - m2) ** 2 + np.trace(X1 + X2 - 2 * C)

        # make sure rounding errors don't cause negative result
        if d < 0:
            d = np.around(d, 4)
        return d if self._squared else np.sqrt(d)


class NormalizedGaussWasserstein(AbstractMetric):
    """
    Implements the Gaussian Wasserstein Score based on the GWD
    """
    def __init__(self, squared=True, norm_mode='gt'):
        """
        Create a new GWS instance for computation of the metric
        :param squared: If the resulting output should be in squared form or not.
        :param norm_mode: one of ["gt", "max", "min"]. The proposed GWS uses "gt", i.e., normalization is performed
        using the ground truth of the object shape. Alternatively the larger ("max") or smaller ("min") of the two
        compared objects can be used for normalization.
        Note that "gt" is the default and suggested parameter. No other choice was used in the corresponding paper.
        """
        self._squared = squared
        self._gwd = GaussWasserstein(squared=squared)

        self._mode = norm_mode
        assert self._mode in ["gt", "max", "min"]

    def get_name(self):
        if self._mode == "gt":
            return f"GWS"
        return f"{self._mode.upper()}-normed-GWD"

    def get_label(self):
        return "Normed Squared GWD" if self._squared else "Normed GWD"

    def calculate(self, m1, p1, m2, p2):
        # area of ellipse = PI * sx1 * sx2 with sx the Semi-aXis lengths
        norm1 = np.pi * p1[1] * p1[2]
        norm2 = np.pi * p2[1] * p2[2]

        if self._mode == "max":
            norm = np.maximum(norm1, norm2)
        elif self._mode == "min":
            norm = np.minimum(norm1, norm2)
        elif self._mode == "gt":
            norm = norm2
        else:
            raise ValueError(f"Unknown norm mode {self._mode}!")

        gwd = self._gwd.calculate(m1, p1, m2, p2)
        return gwd / norm if self._squared else gwd / np.sqrt(norm)
