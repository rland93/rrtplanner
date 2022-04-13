from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.patches import Ellipse
import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D


def remove_axticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def plot_og(ax: Axes, og: np.ndarray, cmap: str = "Greys", vmin=0, vmax=1):
    norm = Normalize(vmin=vmin, vmax=vmax)
    ax.imshow(og.T, cmap=cmap, norm=norm, origin="lower", interpolation=None)


def plot_start_goal(ax, xstart, xgoal):
    ax.scatter(xstart[0], xstart[1], c="r", s=50, label="start", marker="o")
    ax.scatter(xgoal[0], xgoal[1], c="b", s=50, label="goal", marker="*")


def plotly_get_lines(T):
    for e1, e2 in T.edges():
        yield (T.nodes[e1]["pt"], T.nodes[e2]["pt"])


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


def plot_3D_terrain(
    ax: Axes3D,
    X,
    Y,
    Z,
    zsquash: float = 0.5,
    alpha: float = 0.5,
    wireframe: bool = False,
    cmap="magma",
    meshscale=2,
) -> Axes3D:
    ar = X.shape[0] / X.shape[1]
    ax.set_box_aspect((1, 1 * ar, zsquash))
    ax.set_proj_type("ortho")
    zmin, zmax = Z.min(), Z.max()
    norm = Normalize(zmin - abs(zmax - zmin) * 0.1, zmax)
    colors = cm.get_cmap(cmap)(norm(Z))
    if wireframe:
        linewidth = 0.5
    else:
        linewidth = 1.0
    S = ax.plot_surface(
        X,
        Y,
        Z,
        facecolors=colors,
        shade=not wireframe,
        zorder=2,
        rcount=Z.shape[0] // meshscale,
        ccount=Z.shape[1] // meshscale,
        linewidth=linewidth,
    )
    if wireframe:
        S.set_facecolor((0, 0, 0, 0))
    return ax


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
