"""
Implements the KLD for Gaussians
"""
import numpy as np
from numpy.linalg import inv

from src.utilities.utils import to_matrix
from src.metrics.abstract_metric import AbstractMetric


class KullbackLeibler(AbstractMetric):
    """
    Interprets the ellipse as a Gaussian and calculate the Kullback-Leibler divergence. Optionally (symmetric=True), the
    average of the KLD from one ellipse to the other and vice versa can be calculated.
    """
    def __init__(self, squared=True, symmetric=True):
        self._squared = squared
        self._symmetric = symmetric  # use symmetric form

    def get_name(self):
        return "KLD" if self._squared else r"$\sqrt{\mathrm{KLD}}$"

    def get_label(self):
        return self.get_name()

    @staticmethod
    def kullback_leibler(m1, m2, shape1, shape2):
        shape2_inv = inv(shape2)
        return 0.5 * (np.trace(shape2_inv @ shape1) - len(m1) + (m2 - m1) @ shape2_inv @ (m2 - m1)
                      + np.log(np.linalg.det(shape2) / np.linalg.det(shape1)))

    def calculate(self, m1, p1, m2, p2):
        shape1 = to_matrix(np.array(p1))
        shape2 = to_matrix(np.array(p2))

        if self._symmetric:
            kl_difference = 0.5 * (self.kullback_leibler(m1, m2, shape1, shape2)
                                   + self.kullback_leibler(m2, m1, shape2, shape1))
        else:
            kl_difference = self.kullback_leibler(m1, m2, shape1, shape2)

        return kl_difference if self._squared else np.sqrt(kl_difference)
