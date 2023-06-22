"""
Contains function related to the implementation of the OSPA-N metric
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from src.utilities.utils import rot, get_equidistant_ellipse_points
from src.metrics.abstract_metric import AbstractMetric


class OSPA(AbstractMetric):
    def __init__(self,
                 number_of_points=4,
                 squared=True,
                 equidistant=True):
        """
        Create a new OSPA metric instance.

        Points can be sampled in an equidistant fashion on the contour or equally distributed in their angle to the
        center.
        The latter is much faster, but the resulting points will be denser towards the end of the major axis. This
        effect is more pronounced the larger the difference between the semi-axes lengths is.
        :param number_of_points: Number of Points on the contour that should be used
        :param squared: Whether the metric should be squared
        :param equidistant: If True, points will be generated in an equidistant fashion on the contour.
        If False, points are instant equally distributed in their angle to the center.
        """
        self._number_of_points = number_of_points
        self._squared = squared
        self._equidistant_points = equidistant

    def get_name(self):
        # sq = "(non-squared)" if not self._squared else ""
        sq = "(squared)" if self._squared else ""
        return f"OSPA-{self._number_of_points}{sq}"

    def get_label(self):
        return "OSPA-%i / $m^2$" % self._number_of_points if self._squared else "OSPA-%i / $m$" % self._number_of_points

    def calculate(self, m1, p1, m2, p2):
        """
        Calculate OSPA-N for two objects given as
        m: location as [x, y]
        p: shape as [theta, l, w] (orientation theta and semi-axis length l,w)
        :param m1: location as [x, y] of first object
        :param p1: shape as [theta, l, w] (orientation theta and semi-axis length l,w) of first object
        :param m2: location as [x, y] of second object
        :param p2: shape as [theta, l, w] (orientation theta and semi-axis length l,w) of second object
        :return: OSPA-N between both objects
        """
        # get N equally distributed points
        if self._equidistant_points:
            # equidistant points on the contour
            x1_points = get_equidistant_ellipse_points(m1, p1, number_of_points=self._number_of_points)
            x2_points = get_equidistant_ellipse_points(m2, p2, number_of_points=self._number_of_points)
        else:
            # "equiangular" points in angle to center
            theta = np.linspace(0.0, 2.0 * np.pi, self._number_of_points)
            x1_points = (m1[:, None] +
                         rot(p1[0]) @ np.diag([p1[1], p1[2]]) @ np.array([np.cos(theta), np.sin(theta)])).T
            x2_points = (m2[:, None] +
                         rot(p2[0]) @ np.diag([p2[1], p2[2]]) @ np.array([np.cos(theta), np.sin(theta)])).T

        dist_matrix = cdist(x1_points, x2_points) ** 2 if self._squared else cdist(x1_points, x2_points)

        rows, columns = linear_sum_assignment(dist_matrix)

        return np.sum(dist_matrix[rows, columns]) / self._number_of_points
