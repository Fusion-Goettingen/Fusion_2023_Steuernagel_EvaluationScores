"""
Evaluate the effect of increasing the error consistently. Desired behaviour would be for the metric to increase
linearly accordingly.
"""
import numpy as np
import matplotlib.pyplot as plt

from src.metrics.hellinger import GaussianHellinger
from src.metrics.gauss_wasserstein import GaussWasserstein, NormalizedGaussWasserstein
from src.metrics.ospa_contour import OSPA
from src.metrics.intersection_over_union import IntersectionOverUnion
from src.metrics.kullback_leibler import KullbackLeibler
from src.utilities.visuals import plot_elliptic_extent


def create_plot(l, w, delta_error, n_samples,
                metric_list,
                target_dir,
                normalize_with_max,
                metric_linestyles,
                metric_colors,
                metric_markers,
                mark_every):
    # init variables
    zero = np.array([0, 0])
    error_dict = {}

    # generate results
    for metric in metric_list:
        errors = np.array([metric.calculate(zero + d * delta_error[:2],
                                            [0, l, w] + d * delta_error[2:], zero, [0, l, w])
                           for d in range(n_samples)])

        # normalize errors
        if normalize_with_max:
            errors /= np.max(errors)

        # save to dict
        error_dict[metric.get_name()] = errors

    # create plot
    # done twice, once for paper and once for presentation
    for version in ["paper", "presentation"]:
        # determine style based on paper vs pres.
        plt.style.use(f"../../data/stylesheets/{version}.mplstyle")
        for idx, metric in enumerate(metric_list):
            plt.plot(range(n_samples), np.around(error_dict[metric.get_name()], 3), label=metric.get_name(),
                     linestyle=metric_linestyles[idx], color=metric_colors[idx], marker=metric_markers[idx],
                     markevery=mark_every[idx], markersize=16)
        # plt.title(f"Fixed Location Error {offset} for Different Sizes")
        plt.legend(loc="lower right")
        plt.xlabel("Scaling Factor")

        if normalize_with_max:
            plt.ylabel(r"$\frac{\mathrm{Metric\;Result}}{\max\;(\mathrm{Metric\;Result})}$")
            plt.ylabel("Metric Result / max(Metric Result)")
        else:
            plt.ylabel("Metric Result")

        plt.tight_layout()
        # determine save-path based on paper vs pres.
        plt.savefig(target_dir + f"{version}/bounded_metric_delta_behaviour")
        plt.close()

        plt.style.use(f"../../data/stylesheets/{version}.mplstyle")
        plot_elliptic_extent(zero, [0, l, w], fill=True, est_color='grey', label='Ground Truth', color_alpha=0.7)
        plot_elliptic_extent(zero, [0, l, w], fill=False, est_color='blue',
                             label='Closest Estimate')
        plot_elliptic_extent(zero + (n_samples - 1) * 0.5 * delta_error[:2],
                             [0, l, w] + (n_samples - 1) * 0.5 * delta_error[2:], fill=False,
                             est_color='orange', label='Middle Estimate')
        plot_elliptic_extent(zero + (n_samples - 1) * delta_error[:2],
                             [0, l, w] + (n_samples - 1) * delta_error[2:], fill=False,
                             est_color='red',
                             label='Furthest Estimate')
        plt.arrow(zero[0] + (n_samples - 1) * 0.1 * delta_error[0],
                  zero[1] + (n_samples - 1) * 0.1 * delta_error[1],
                  (n_samples - 1) * 0.25 * delta_error[0],
                  (n_samples - 1) * 0.25 * delta_error[1], head_width=0.1, color='black')
        plt.arrow(zero[0] + (n_samples - 1) * 0.6 * delta_error[0],
                  zero[1] + (n_samples - 1) * 0.6 * delta_error[1],
                  (n_samples - 1) * 0.25 * delta_error[0],
                  (n_samples - 1) * 0.25 * delta_error[1], head_width=0.1, color='black')

        plt.xlabel("$m_1$ / m")
        plt.ylabel("$m_2$ / m")

        plt.axis('equal')
        plt.tight_layout()
        plt.legend()
        plt.savefig(target_dir + f"{version}/bounded_metric_delta_behaviour_example")
        plt.close()


def main():
    # EXPERIMENT HYPERPARAMETERS
    obj_base_length = 2
    obj_base_width = 1
    n_samples = 100  # number of object generated from base_error + delta_error
    delta_error = np.array([1.0, 1.0, 0.0, 1.0, 1.0]) * 5.0 / n_samples  # constant change of error
    normalize_with_max = True  # whether to divide all metric results by their respective maximum

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
    metric_linestyles = [
        'solid', 'solid', 'solid', 'solid', 'solid', 'solid'
    ]
    metric_colors = [
        'magenta', 'blue', 'orange', 'green', 'red', 'brown'
    ]
    metric_markers = [
        'o', 'P', '*', 'X', 's', '^'
    ]
    # mark_every argument tuples for metrics, for offsetting markers against each other:
    #   (offset, mark_every): (float, float)
    #   small offset for different start times, 1/mark_every = n_markers along x axis
    mark_every = [
        (i * 0.02, 0.1) for i in range(len(metrics))
    ]

    create_plot(l=obj_base_length,
                w=obj_base_width,
                delta_error=delta_error,
                n_samples=n_samples,
                metric_list=metrics,
                target_dir=target_dir,
                normalize_with_max=normalize_with_max,
                metric_linestyles=metric_linestyles,
                metric_colors=metric_colors,
                metric_markers=metric_markers,
                mark_every=mark_every
                )


if __name__ == '__main__':
    main()
