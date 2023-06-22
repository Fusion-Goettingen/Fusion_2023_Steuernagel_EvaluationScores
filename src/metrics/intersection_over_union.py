"""
Contains function related to the implementation of the Intersection-over-Union and its generalized version
(Rezatofighi, Hamid, et al. "Generalized intersection over union: A metric and a loss for bounding box regression."
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.)
"""
import numpy as np
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull

from src.utilities.utils import rot, area_polygon
from src.metrics.abstract_metric import AbstractMetric


class IntersectionOverUnion(AbstractMetric):
    """Computation of (G)IoU using polygon approximations"""
    def __init__(self, generalized=False, number_of_points=100, flip=False):
        """
        Create an instance of the IoU metric.
        :param generalized: If True, calculate GIoU instead of IoU
        :param number_of_points: Number of points used to approximate the ellipses with for the polygon IoU calculatin
        :param flip: If True, will return 1-(G)IoU instead, s.t. higher values indicate higher error rather than
        better overlap.
        """
        self._generalized = generalized
        self._number_of_points = number_of_points
        self._flip = flip

    def get_name(self):
        return self.get_label()
        # return "IoU" if not self._generalized else "GIoU"

    def get_label(self):
        if self._flip:
            flip = r"1 - "
        else:
            flip = ""
        return f"{flip}IoU" if not self._generalized else f"{flip}GIoU"

    def calculate(self, m1, p1, m2, p2):
        """
        Calculate IoU or GIoU for two objects given as
        m: location as [x, y]
        p: shape as [theta, l, w] (orientation theta and semi-axis length l,w)
        :param m1: location as [x, y] of first object
        :param p1: shape as [theta, l, w] (orientation theta and semi-axis length l,w) of first object
        :param m2: location as [x, y] of second object
        :param p2: shape as [theta, l, w] (orientation theta and semi-axis length l,w) of second object
        :return: IoU or GIoU between both objects
        """
        # get points on ellipses
        theta = np.linspace(0.0, 2.0 * np.pi, self._number_of_points)
        m1, m2, p1, p2 = np.array(m1), np.array(m2), np.array(p1), np.array(p2)
        x1_points = m1[:, None] + rot(p1[0]) @ np.diag([p1[1], p1[2]]) @ np.array([np.cos(theta), np.sin(theta)])
        x2_points = m2[:, None] + rot(p2[0]) @ np.diag([p2[1], p2[2]]) @ np.array([np.cos(theta), np.sin(theta)])

        # create polygon
        x1_pol = Polygon(x1_points.T)
        x2_pol = Polygon(x2_points.T)

        # calculate IoU
        intersection = x2_pol.intersection(x1_pol).area
        union = x2_pol.area + x1_pol.area - intersection
        iou = intersection / union

        if self._generalized:
            # calculate GIoU
            joint_points = np.vstack([x1_points.T, x2_points.T])
            hull = ConvexHull(joint_points)
            c_area = area_polygon(joint_points[hull.vertices])
            iou = iou - (c_area - union) / c_area

        if self._flip:
            iou = 1 - iou

        return iou
