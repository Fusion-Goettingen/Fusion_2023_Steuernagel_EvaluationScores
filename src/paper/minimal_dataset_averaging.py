"""
Minimal simulation of a data set for which a scalar overall error is calculated based on different estimates.
The data set consists of two differently-sized objects, and the errors made by the estimates vary only in rotation.
One of the two objects is estimated well, the other one has a larger error.

The desired result (if object size should not be accounted for in the average) is for a metric to produce equivalent
results for both estimates.
"""
import numpy as np
import matplotlib.pyplot as plt

from src.metrics.hellinger import GaussianHellinger
from src.metrics.gauss_wasserstein import GaussWasserstein, NormalizedGaussWasserstein
from src.metrics.ospa_contour import OSPA
from src.metrics.intersection_over_union import IntersectionOverUnion
from src.metrics.kullback_leibler import KullbackLeibler
from src.utilities.visuals import plot_elliptic_extent


def create_plot(metric_dict,
                target_dir):
    # location and shape of the objects
    small_center = np.array([6, -4])
    small_shape = np.array([np.pi / 2, 4, 2])
    large_center = np.array([-6, 0])
    large_shape = np.array([np.pi / 2, 8, 4])

    # rotation errors and color to be used
    error_dict_list = [
        {
            "s": np.pi / 2,
            "l": np.pi / 4,
            'c': "blue"
        },
        {
            "s": np.pi / 4,
            "l": np.pi / 2,
            "c": 'red'
        }
    ]

    # main function kernel
    def draw(version, include_print=False):
        """
        Internal function to create the plot, wrapped as function so that it can be called multiple times for
        different styles
        """
        # plot ground truth
        plot_elliptic_extent(small_center, small_shape, est_color='grey', label="Ground Truth", fill=True,
                             color_alpha=0.7)
        plot_elliptic_extent(large_center, large_shape, est_color='grey', label=None, fill=True, color_alpha=0.7)

        # plot scenarios
        for i, error_dict in enumerate(error_dict_list):
            small_error = small_shape + [error_dict["s"], 0, 0]
            large_error = large_shape + [error_dict["l"], 0, 0]

            color = error_dict['c']

            plot_elliptic_extent(small_center, small_error, est_color=color, label=f"Setting {i + 1}")
            plot_elliptic_extent(large_center, large_error, est_color=color, label=None)
            if include_print:
                print(f"Setting {i + 1} ({color}):")
                for metric in metric_dict:
                    small_d = metric.calculate(small_center, small_error, small_center, small_shape)
                    large_d = metric.calculate(large_center, large_error, large_center, large_shape)
                    avg_d = 0.5 * (small_d + large_d)
                    output_str = f"{metric.get_name()}: {avg_d:.2f}"
                    print("\t" + output_str)

        # show
        plt.legend()
        plt.axis('equal')
        plt.xlabel("$m_1$ / m")
        plt.ylabel("$m_2$ / m")
        plt.tight_layout()
        plt.savefig(target_dir + f"{version}/minimal_dataset_averaging")
        plt.close()

    # draw
    for i, version in enumerate(["paper", "presentation"]):
        plt.style.use(f"../../data/stylesheets/{version}.mplstyle")
        draw(version=version, include_print=(i == 0))


def main():
    # SAVE TO
    target_dir = "../../figures/"

    # EVALUATED METRICS
    metrics = [
        GaussWasserstein(squared=False),
        IntersectionOverUnion(generalized=True, flip=True),
        OSPA(number_of_points=100, squared=False),
        NormalizedGaussWasserstein(squared=False),
        KullbackLeibler(squared=False, symmetric=False),
        GaussianHellinger(squared=False)
    ]
    # run
    create_plot(metric_dict=metrics,
                target_dir=target_dir)


if __name__ == '__main__':
    main()
