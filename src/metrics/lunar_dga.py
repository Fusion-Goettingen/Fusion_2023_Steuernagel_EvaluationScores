"""
Implements the distance proposed in
Christian, J.A., Derksen, H. & Watkins, R. Lunar Crater Identification in Digital Images.
J Astronaut Sci 68, 1056–1144 (2021). https://doi.org/10.1007/s40295-021-00287-8

Link: https://link.springer.com/article/10.1007/s40295-021-00287-8
"""
import numpy as np
from numpy.linalg import inv, det

from src.utilities.utils import rot
from src.metrics.abstract_metric import AbstractMetric


class LunarDga(AbstractMetric):
    def __init__(self, normalized=False):
        self._normalized = normalized

    @staticmethod
    def _to_matrix(p):
        return rot(p[0]) @ np.diag(p[1:]) @ rot(p[0]).T

    def calculate(self, m1, p1, m2, p2):
        Yi = self._to_matrix(p1)
        Yj = self._to_matrix(p2)
        yi = np.array(m1[:2])
        yj = np.array(m2[:2])
        prefix_factor = (4 * np.sqrt(det(Yi) * det(Yj))) / (det(Yi + Yj))
        y_diff = yi - yj
        exponent = -0.5 * (y_diff.T @ Yi @ inv(Yi + Yj) @ Yj @ y_diff)
        v = prefix_factor * np.exp(exponent)
        # rounding errors can pose problems, ensure arccos input is in [-1, 1] via clipping
        v = np.clip(v, -1, 1)
        if not self._normalized:
            return np.arccos(v)
        else:
            # adapt accordingly, using sigma_img = 1
            norm = 0.85 / np.sqrt(p2[1] * p2[2])
            return np.arccos(v) / norm

    def get_name(self):
        if not self._normalized:
            return "Dga"
        else:
            return "Dga(norm)"

    def get_label(self):
        return self.get_name()
