from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.patches import Ellipse
import numpy as np


def remove_axticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def plot_world(ax, world, cmap="Greys", vmin=0, vmax=1):
    norm = Normalize(vmin=vmin, vmax=vmax)
    ax.imshow(world.T, cmap=cmap, norm=norm, origin="lower", interpolation=None)


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
        colors = cm.get_cmap("viridis")(norm(costs))
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


def plot_path(ax, pathlines: np.ndarray):
    lc = LineCollection(pathlines, color="r", linewidth=2)
    ax.add_collection(lc)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from world_gen import make_world, get_rand_start_end
    from rrt import RRTStandard, RRTStar, RRTStarInformed

    axs = []

    # create world
    w, h = 256, 256
    n = 900
    r_rewire, r_goal = 64, 32
    world = make_world((w, h), (4, 4))
    world = world | make_world((w, h), (2, 2))
    xstart, xgoal = get_rand_start_end(world)

    # world plots
    fig1, ax1 = plt.subplots(dpi=200)
    axs.append(ax1)
    # # RRT plots
    fig2, ax2 = plt.subplots(dpi=200)
    axs.append(ax2)

    rrt2 = RRTStandard(world, n)
    T, gv = rrt2.make(xstart, xgoal)
    plot_rrt_lines(ax2, T)

    # # RRT* plots
    fig3, ax3 = plt.subplots(dpi=200)
    axs.append(ax3)

    rrt3 = RRTStar(world, n, r_rewire)
    T, gv = rrt3.make(xstart, xgoal)
    plot_rrt_lines(ax3, T)
    path = rrt3.path_points(T, rrt3.route2gv(T, gv))
    plot_path(ax3, path)

    # # RRT* Informed plots
    fig4, ax4 = plt.subplots(dpi=200)
    axs.append(ax4)

    rrt4 = RRTStarInformed(world, n, r_rewire, r_goal)
    T, gv = rrt4.make(xstart, xgoal)
    plot_rrt_lines(ax4, T)
    plot_ellipses(ax4, rrt4.ellipses)
    plot_path(ax4, path)

    for ax in axs:
        remove_axticks(ax)
        plot_world(ax, world)
        plot_start_goal(ax, xstart, xgoal)

    plt.show()
