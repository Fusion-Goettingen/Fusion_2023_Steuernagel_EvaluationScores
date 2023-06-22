"""
Implements the Hellinger Distance between Gaussians.
"""
import numpy as np
from numpy.linalg import inv

from src.utilities.utils import rot
from src.metrics.abstract_metric import AbstractMetric


class GaussianHellinger(AbstractMetric):
    def __init__(self, squared=True):
        self._squared = squared

    def get_name(self):
        return "Hellinger"

    def get_label(self):
        return "Squared Hellinger" if self._squared else "Hellinger"

    def calculate(self, m1, p1, m2, p2):
        """
        Calculate Gaussian Hellinger Distance for two objects given as
        m: location as [x, y]
        p: shape as [theta, l, w] (orientation theta and semi-axis length l,w)

        c.f. "Statistical Inference based on Divergence Measured" by Leandro Pardo Llorente
        :param m1: location as [x, y] of first object
        :param p1: shape as [theta, l, w] (orientation theta and semi-axis length l,w) of first object
        :param m2: location as [x, y] of second object
        :param p2: shape as [theta, l, w] (orientation theta and semi-axis length l,w) of second object
        :return: (Squared) Gaussian Hellinger distance between both objects
        """
        m1, p1, m2, p2 = np.array(m1), np.array(p1), np.array(m2), np.array(p2)
        S1 = rot(p1[0]) @ np.diag(p1[1:] ** 2) @ rot(p1[0]).T
        S2 = rot(p2[0]) @ np.diag(p2[1:] ** 2) @ rot(p2[0]).T
        S_mean = 0.5 * S1 + 0.5 * S2

        u = (np.array(m1) - np.array(m2)).reshape((-1, 1))
        frac = np.linalg.det(S1) ** 0.25 * np.linalg.det(S2) ** 0.25
        frac /= np.linalg.det(S_mean) ** 0.5

        d = 1 - frac * np.exp(-(1 / 8) * u.T @ np.linalg.inv(S_mean) @ u)
        d = np.asarray(d).round(8).flatten()[0]  # rounding errors can cause negative values -> round to 8 decimals
        return d if self._squared else np.sqrt(d)
