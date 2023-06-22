"""
Implements general utility functions
"""
import numpy as np
from scipy.optimize import root
from scipy.special import ellipeinc


def rot(theta):
    """
    Constructs a rotation matrix for given angle alpha.
    :param theta: angle of orientation
    :return: Rotation matrix in 2D around theta (2x2)
    """
    r = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return r.reshape((2, 2))


def area_polygon(points):
    """
    Calculates the area of a polygon
    :param points:  ordered set of points
    :return:        area
    """
    return 0.5 * abs(
        np.sum([points[i, 0] * points[(i + 1) % len(points), 1] - points[(i + 1) % len(points), 0] * points[i, 1]
                for i in range(len(points))]))


def to_matrix(p, square_root=False):
    """
    Transforms shape parameter into matrix.
    :param p:           shape state consisting of [orientation, semi-axis length, semi-axis width]
    :param square_root: set to true if square root should be calculated instead
    :return:            matrix or square root of matrix
    """
    sr = 1 if square_root else 2
    rmat = rot(p[0])
    return rmat @ np.diag(p[1:3] ** sr) @ rmat.T


def to_ellipse_parameters(shape, square_root=False):
    """
    Transforms matrix into shape parameters, i.e., orientation, semi-axis length and semi-axis width
    :param shape:       2x2 shape matrix
    :param square_root: set to true if the input shape is actually the square root matrix
    :return:            orientation, semi-axis length, semi-axis width
    """
    vals, vecs = np.linalg.eig(shape @ shape) if square_root else np.linalg.eig(shape)
    return np.array([np.arctan2(vecs[1, 0], vecs[0, 0]), np.sqrt(vals[0]), np.sqrt(vals[1])]) if len(vecs.shape) == 2 \
        else np.array([np.arctan2(vecs[:, 1, 0], vecs[:, 0, 0]), np.sqrt(vals[:, 0]), np.sqrt(vals[:, 1])]).T


def equidistant_angles_in_ellipse(num, a, b):
    """
    Get angles of num points on an ellipse contour with minor axis length a and major axis length b
    https://stackoverflow.com/a/52062369
    """
    assert (num > 0)
    assert (a <= b), "Semi Axis length a needs to be smaller than b!"
    angles = 2 * np.pi * np.arange(num) / num
    if a != b:
        e2 = (1.0 - a ** 2.0 / b ** 2.0)
        tot_size = ellipeinc(2.0 * np.pi, e2)
        arc_size = tot_size / num
        arcs = np.arange(num) * arc_size
        res = root(lambda x: (ellipeinc(x, e2) - arcs), angles)
        angles = res.x
    return angles


def get_equidistant_ellipse_points(m, p, number_of_points=100):
    """
    Get equally spaced points on the contour of an ellipse

    :param m: Center of ellipse as 2D array
    :param p: [Orientation, length, width] of ellipse
    :param number_of_points: Number of points to sample
    :return: Points with equidistant spacing on ellipse contour as ndarray of shape
    """
    # prepare
    m = np.array(m).astype(float)
    p = np.array(p).astype(float)

    # angles_in_ellipse requires p[1] < p[2]
    # if this is not the case, flip them, which corresponds to a 90° rotation
    if p[1] >= p[2]:
        # flip
        p[1:] = p[1:][::-1]
        # mark in angle as rotation
        p[0] = (p[0] + np.pi / 2) % (2 * np.pi)

    # calculate angles
    theta = equidistant_angles_in_ellipse(number_of_points, *p[1:])
    # get points for angles
    points = m[:, None] + rot(p[0]) @ np.diag([p[1], p[2]]) @ np.array([np.cos(theta), np.sin(theta)])
    return points.T
