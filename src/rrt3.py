from rrtplanner import rrt, oggen, surface, plots
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from math import sqrt
from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import Normalize
from matplotlib import cm


def product(*arrays):
    mesh = np.meshgrid(*arrays)  # standard numpy meshgrid
    dim = len(mesh)  # number of dimensions
    elements = mesh[0].size  # number of elements, any index will do
    flat = np.concatenate(mesh).ravel()  # flatten the whole meshgrid
    reshape = np.reshape(flat, (dim, elements)).T  # reshape and transpose
    return reshape


def get_grid_interp(X, Y, S):
    xs = X[0, :]
    ys = Y[:, 0]
    return RegularGridInterpolator((xs, ys), S.T)


def get_layout_3d(T, X, Y, S, n=10):
    gridinterp = get_grid_interp(X, Y, S)
    lines, costs = [], []
    for e1, e2 in T.edges():
        # unpack points from i, j -> x, y
        p1i = T.nodes[e1]["pt"][0]
        p1j = T.nodes[e1]["pt"][1]
        p2i = T.nodes[e2]["pt"][0]
        p2j = T.nodes[e2]["pt"][1]
        p1 = np.array([X[p1i, p1j], Y[p1i, p1j]])
        p2 = np.array([X[p2i, p2j], Y[p2i, p2j]])
        # interpolate (x, y) -> z and create 3-D array
        linex = np.linspace(p1, p2, num=n)
        linez = np.atleast_2d(gridinterp(linex)).T
        line3d = np.concatenate((linex, linez), axis=1)
        lines.append(line3d)
        # append cost of the line
        costs.append(T.edges[e1, e2]["cost"])
    return lines, costs


if __name__ == "__main__":
    from plotly import graph_objects as go
    import plotly.offline as pyo

    WIDTH, HEIGHT, DEPTH = 120, 80, 25
    aspectyx = float(HEIGHT) / WIDTH
    aspectzx = float(DEPTH) / HEIGHT
    xmax, ymax = WIDTH, HEIGHT
    cols, rows = WIDTH, HEIGHT
    gaph, minh, maxdx, maxd2x = 3.5, 2.5, 0.95, 0.07
    X, Y, H = oggen.example_terrain(xmax=xmax, ymax=ymax, cols=cols, rows=rows)
    H *= float(DEPTH) / H.ptp()
    waypoints = surface.generate_example_waypoints(
        X,
        Y,
        H,
        2,
        (10.0, 35.0),
    )

    # "Low Surface"
    surf2 = surface.SurfaceLowHeight(X, Y, H)
    surf2.setup(
        minh,
        gaph,
        maxdx,
        maxd2x,
        use_parameters=False,
    )
    surf2.solve(verbose=True, solver="GUROBI")
    for k, v in surf2.objectives.items():
        print(f"{k}: {v.value}")

    alpha = 5.0
    beta = 2.0
    gridinterp = get_grid_interp(X, Y, surf2.S.value)

    def cost3d(
        vcosts: np.ndarray,
        points: np.ndarray,
        v: int,
        x: np.ndarray,
    ) -> float:

        if points[v][0] == -1 or points[v][1] == -1:
            return -1

        i0, j0 = points[v]
        i1, j1 = x
        p0 = np.array([X[i0, j0], Y[i0, j0]])
        p1 = np.array([X[i1, j1], Y[i1, j1]])
        # d = np.linalg.norm(p1 - p0)
        # ps = np.linspace(p0, p1, num=int(d))
        # zs = gridinterp(ps, method="nearest")
        # zsum = zs.sum() / int(d)
        d = sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
        zcost = alpha * 0.0
        xcost = beta * d
        return vcosts[v] + zcost + xcost

    rrts = rrt.RRTStar(
        np.zeros_like(X),
        1000,
        r_rewire=64.0,
        costfn=cost3d,
    )
    xstart = np.array([1, 1])
    xgoal = np.array([35, 25])

    T, gv = rrts.plan(xstart, xgoal)

    # lines, costs = get_layout_3d(T, X, Y, surf2.S.value)
    # norm = Normalize(vmin=min(costs), vmax=max(costs))
    # cmap = cm.get_cmap("coolwarm")
    # colors = cmap(norm(costs))
    # fig = go.Figure()
    # for line, color in zip(lines, colors):
    #     fig.add_trace(
    #         go.Scatter3d(
    #             x=line[:, 0],
    #             y=line[:, 1],
    #             z=line[:, 2],
    #             mode="lines",
    #             line=dict(
    #                 width=1.0,
    #                 color="black",
    #             ),
    #         ),
    #     )

    # fig.update_traces(showlegend=False)
    # pyo.plot(fig)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    plots.plot_rrt_lines(ax, T, color_costs=True)
    ax.contour(Y, X, surf2.S.value, levels=12)
    # plt.show()
