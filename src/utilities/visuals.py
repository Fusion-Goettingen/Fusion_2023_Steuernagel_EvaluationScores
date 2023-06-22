import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_elliptic_extent(m, p, ax=None, est_color='b', color_alpha=1., label="Extent Estimate", linestyle=None,
                         show_center=True, fill=False):
    """
    Add matplotlib ellipse patch based on location and extent information about vehicle
    :param m: Kinematic information as 4D array [x, y, velocity_x, velocity_y]
    :param p: extent information as 3D array [orientation, semi-length, semi-width]. Orientation in radians.
    :param ax: matplotlib axis to plot on or None (will use .gca() if None)
    :param est_color: Color to plot the ellipse and marker in
    :param color_alpha: Alpha value for plot
    :param label: Label to apply to plot or None to not add a label
    :param linestyle: Linestyle parameter passed to matplotlib
    :param show_center: If True, will additionally add an x for the center location
    :param fill: If true, ellipse will be filled
    """
    if ax is None:
        ax = plt.gca()
    alpha, l1, l2 = p
    alpha = np.rad2deg(alpha)
    # patches.Ellipse takes angle counter-clockwise
    el = patches.Ellipse(xy=m[:2], width=2.0*l1, height=2.0*l2, angle=alpha, fill=fill, color=est_color, label=label,
                         alpha=color_alpha, linestyle=linestyle)
    if show_center:
        ax.scatter(m[0], m[1], color=est_color, marker='x')
    ax.add_patch(el)
