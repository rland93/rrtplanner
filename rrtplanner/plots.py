from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.patches import Ellipse
import numpy as np
from matplotlib.axes import Axes


def remove_axticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def plot_og(ax: Axes, og: np.ndarray, cmap: str = "Greys", vmin=0, vmax=1):
    norm = Normalize(vmin=vmin, vmax=vmax)
    ax.imshow(og.T, cmap=cmap, norm=norm, origin="lower", interpolation=None)


def plot_start_goal(ax, xstart, xgoal):
    ax.scatter(xstart[0], xstart[1], c="r", s=50, label="start", marker="o")
    ax.scatter(xgoal[0], xgoal[1], c="b", s=50, label="goal", marker="*")


def plot_rrt_lines(ax, T, color_costs=True, cmap="viridis", color="tan", alpha=1.0):
    lines = []
    costs = []
    for e1, e2 in T.edges():
        lines.append((T.nodes[e1]["pt"], T.nodes[e2]["pt"]))
        costs.append(T.edges[e1, e2]["cost"])
    if color_costs:
        norm = Normalize(vmin=min(costs), vmax=max(costs))
        colors = cm.get_cmap(cmap)(norm(costs))
        lc = LineCollection(lines, colors=colors, alpha=alpha)
    else:
        lc = LineCollection(lines, color=color, alpha=alpha)
    ax.add_collection(lc)


def plot_rrt_points(ax, T):
    for n in T.nodes:
        ax.scatter(T.nodes[n]["pt"][0], T.nodes[n]["pt"][1], c="k", s=10, marker="o")


def plot_ellipses(ax, ellipses, cmap="RdPu"):
    if len(ellipses) == 0:
        return None
    colors = [j for j in ellipses.keys()]
    norm = Normalize(vmin=min(colors), vmax=max(colors))
    colors = cm.get_cmap(cmap)(norm(colors))
    for e, c in zip(ellipses.values(), colors):
        ax.add_artist(Ellipse(*e, fill=None, color=c, zorder=10))


def plot_path(ax, pathlines: np.ndarray, zorder=10):
    lc = LineCollection(pathlines, color="r", linewidth=2, zorder=zorder)
    ax.add_collection(lc)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import rrt, oggen

    axs = []
    dpi, figsize = 120, (12, 6)

    # create world
    w, h = 700, 500
    n = 2500
    r_rewire, r_goal = 64, 32
    og = oggen.perlin_occupancygrid(w, h)
    xstart = rrt.random_point_og(og)
    xgoal = rrt.random_point_og(og)

    fig, axs = plt.subplots(2, 2, figsize=figsize, dpi=dpi, tight_layout=True)

    axs[0, 0].set_title("Empty OccupancyGrid")

    rrt2 = rrt.RRTStandard(og, n)
    T, gv = rrt2.make(xstart, xgoal)
    path = rrt2.path_points(T, rrt2.route2gv(T, gv))
    plot_rrt_lines(axs[0, 1], T)
    plot_path(axs[0, 1], path)
    axs[0, 1].set_title("Standard RRT")

    rrt3 = rrt.RRTStar(og, n, r_rewire)
    T, gv = rrt3.make(xstart, xgoal)
    plot_rrt_lines(axs[1, 0], T)
    path = rrt3.path_points(T, rrt3.route2gv(T, gv))
    plot_path(axs[1, 0], path)
    axs[1, 0].set_title("RRT*")

    rrt4 = rrt.RRTStarInformed(og, n, r_rewire, r_goal)
    T, gv = rrt4.make(xstart, xgoal)
    path = rrt4.path_points(T, rrt4.route2gv(T, gv))
    plot_rrt_lines(axs[1, 1], T)
    plot_ellipses(axs[1, 1], rrt4.ellipses)
    plot_path(axs[1, 1], path)
    axs[1, 1].set_title("RRT* Informed")

    for ax in axs.flatten():
        remove_axticks(ax)
        plot_og(ax, og)
        plot_start_goal(ax, xstart, xgoal)

    # fig.savefig("./figure.png", dpi=200)

    plt.show()
