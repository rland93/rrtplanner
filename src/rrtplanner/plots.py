from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.patches import Ellipse
import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D
from plotly import graph_objects as go
from plotly import subplots
from matplotlib import gridspec


def plot_surface(ax: Axes3D, X, Y, S, zsquash=0.2, wireframe=True, cmap="viridis"):
    if not isinstance(ax, Axes3D):
        raise TypeError(f"ax must be an Axes3D object. Got {type(ax)}")

    ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(S) * zsquash))
    # ax.set_proj_type("ortho")
    # draw sheet
    if wireframe:
        hmin, hmax = S.min(), S.max()
        norm = Normalize((hmin - hmax) * 0.03, hmax)
        colors = cm.get_cmap(cmap)(norm(S))
        s = ax.plot_surface(
            X,
            Y,
            S,
            zorder=2,
            linewidths=0.5,
            shade=False,
            facecolors=colors,
            rcount=X.shape[0],
            ccount=X.shape[1],
        )
        s.set_facecolor((0, 0, 0, 0))
    else:
        ax.plot_surface(X, Y, S, cmap=cm.get_cmap(cmap), zorder=2)
    return ax


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


def make_ensemble_matplotlib(
    X,
    Y,
    H,
    S,
    waypoints=None,
):
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(3, 2, figure=fig)

    ax1 = fig.add_subplot(gs[:2, :], projection="3d")
    plot_3D_terrain(ax1, X, Y, H, cmap="gist_earth")
    plot_3D_terrain(ax1, X, Y, S, wireframe=True, cmap="bone")
    if waypoints is not None:
        ax1.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], "o", c="r")
    ax1.set_title("3-D View")

    ax2 = fig.add_subplot(gs[2, 0])
    ax2.contour(X, Y, H, levels=10, cmap="gist_earth")
    if waypoints is not None:
        ax2.plot(waypoints[:, 0], waypoints[:, 1], "o", c="r")
    ax2.set_title("2-D Contur (Terrain)")
    ax2.set_aspect("equal")

    ax3 = fig.add_subplot(gs[2, 1])
    ax3.set_title("2-D Contur (Surface)")
    ax3.contour(X, Y, S, levels=20, cmap="bone")
    if waypoints is not None:
        ax3.plot(waypoints[:, 0], waypoints[:, 1], "o", c="r")

    ax3.set_aspect("equal")
    return fig


def generate_ensemble_plotly(
    X,
    Y,
    H,
    S,
    aspectyx,
    aspectzx,
    waypoints=None,
    figtitle="Terrain",
    ax3d_title="3D View",
    ax2d_terrain_title="Contour (Terrain)",
    ax2d_surface_title="Contour (Surface)",
    terrain_cmap="gray_r",
    surface_cmap="Ice",
):
    fig = subplots.make_subplots(
        rows=2,
        cols=2,
        specs=[[{"rowspan": 2, "is_3d": True}, {}], [None, {}]],
        subplot_titles=(ax3d_title, ax2d_terrain_title, ax2d_surface_title),
    )
    # Terrain mesh
    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=H,
            colorscale=terrain_cmap,
            colorbar=dict(x=0.0),
            showscale=False,
        ),
        row=1,
        col=1,
    )
    # Surface mesh
    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=S,
            opacity=0.45,
            colorscale=surface_cmap,
            colorbar=dict(x=0.05),
            showscale=False,
        ),
        row=1,
        col=1,
    )
    # 3D waypoints
    if waypoints is not None:
        fig.add_trace(
            go.Scatter3d(
                x=waypoints[:, 0],
                y=waypoints[:, 1],
                z=waypoints[:, 2],
                mode="markers",
                marker=dict(color="blue", size=5),
            ),
            row=1,
            col=1,
        )

    # 2D Contour, Terrain
    fig.add_trace(
        go.Contour(
            z=H,
            contours_coloring="lines",
            colorscale=terrain_cmap,
            ncontours=20,
            showscale=False,
            showlegend=False,
            line_width=2,
        ),
        row=1,
        col=2,
    )
    # 2D Waypoints
    if waypoints is not None:
        for (r, c) in [(1, 2), (2, 2)]:
            fig.add_trace(
                go.Scatter(
                    x=waypoints[:, 0],
                    y=waypoints[:, 1],
                    mode="markers",
                    marker=dict(color="blue", size=5),
                    showlegend=False,
                ),
                row=r,
                col=c,
            )
    # 2D Contour of Surface
    fig.add_trace(
        go.Contour(
            z=S,
            contours_coloring="lines",
            colorscale=surface_cmap,
            ncontours=20,
            showscale=False,
            showlegend=False,
            line_width=2,
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title="Terrain",
        margin=dict(
            l=65,
            r=50,
            b=65,
            t=90,
        ),
    )

    fig.update_layout(
        scene_aspectmode="manual",
        scene_aspectratio=dict(
            x=1.0,
            y=aspectyx,
            z=aspectzx,
        ),
    )
    # fig.update_yaxes(col=2, row=1, scaleanchor="x", scaleratio=1)
    fig.update_yaxes(col=2, row=2, scaleanchor="x", scaleratio=1)
    return fig


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
