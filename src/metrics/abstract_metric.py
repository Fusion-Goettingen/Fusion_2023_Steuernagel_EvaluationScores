from abc import ABC, abstractmethod


class AbstractMetric(ABC):
    @abstractmethod
    def calculate(self, m1, p1, m2, p2):
        """
        Calculate distance between two objects given as
        m: location as [x, y]
        p: shape as [theta, l, w] (orientation theta and semi-axis length l,w)
        :param m1: location as [x, y] of first object
        :param p1: shape as [theta, l, w] (orientation theta and semi-axis length l,w) of first object
        :param m2: location as [x, y] of second object
        :param p2: shape as [theta, l, w] (orientation theta and semi-axis length l,w) of second object
        :return: Squared distance between both objects
        """
        pass

    @abstractmethod
    def get_name(self):
        """
        :return: name of the metric
        """

    @abstractmethod
    def get_label(self):
        """
        :return: label of y-axis for plotting
        """

    def calculate_uncertain(self, m1, p1, cov1, m2, p2, cov2):
        """
        Calculate distance between two objects given as
        m: location as [x, y]
        p: shape as [theta, l, w] (orientation theta and semi-axis length l,w)
        cov: Covariance as 5x5 array

        By default, this just ignores the covariance entries and passes the call on to self.calculate(...)
        :param m1: location as [x, y] of first object
        :param p1: shape as [theta, l, w] (orientation theta and semi-axis length l,w) of first object
        :param cov1: covariance of first object in order [x, y, theta, l, w]
        :param m2: location as [x, y] of second object
        :param p2: shape as [theta, l, w] (orientation theta and semi-axis length l,w) of second object
        :param cov2: covariance of second object in order [x, y, theta, l, w]
        :return: Squared distance between both objects
        """
        return self.calculate(m1, p1, m2, p2)
