from rrtplanner import rrt, oggen, surface, plots
import numpy as np
from scipy.interpolate import RegularGridInterpolator


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


def get_colorstr(color):
    colorstr = "rgb(" + str(int(color[0] * 255)) + ","
    colorstr += str(int(color[1] * 255)) + ","
    colorstr += str(int(color[2] * 255)) + ")"
    return colorstr


if __name__ == "__main__":
    from plotly import offline as pyo
    from plotly import graph_objects as go
    from matplotlib import colors, cm
    import pathlib, os

    PLOT_DIR = pathlib.Path(__file__).parent.resolve()
    PLOT_DIR = PLOT_DIR / "../../surfaceplanner/urop/"
    os.makedirs(PLOT_DIR / "figures", exist_ok=True)
    print("Saving plots to ", PLOT_DIR)

    W, H, D = 100, 100, 25
    xmax, ymax = W, H
    cols, rows = W, H
    gaph, minh, maxdx, maxd2x = 3.5, 2.5, 0.95, 0.07
    X, Y, Z = oggen.example_terrain(xmax=xmax, ymax=ymax, cols=cols, rows=rows)
    Z *= float(D) / Z.ptp()
    waypoints = surface.generate_example_waypoints(
        X,
        Y,
        Z,
        2,
        (10.0, 35.0),
    )

    # "Low Surface"
    surf = surface.SurfaceLowHeight(X, Y, Z)
    surf.setup(
        minh,
        gaph,
        maxdx,
        maxd2x,
        use_parameters=False,
    )
    surf.solve(verbose=True, solver="GUROBI")
    for k, v in surf.objectives.items():
        print(f"{k}: {v.value}")
    S = surf.S.value
    gridinterp = get_grid_interp(X, Y, S)

    alpha = 5.0
    beta = 2.0

    def cost3d(
        vcosts: np.ndarray,
        points: np.ndarray,
        v: int,
        x: np.ndarray,
    ) -> float:

        if points[v][0] == -1 or points[v][1] == -1:
            return -1

        # line between points
        linepts = surface.linedraw(points[v], x)
        # distance between points
        d = np.linalg.norm(points[v] - x)
        # average height
        s = 0
        for lp in linepts:
            s += S[lp[0], lp[1]]
        s /= float(linepts.shape[0])
        # we have R2 xy cost and height cost
        # as separate components
        zcost = alpha * s
        xcost = beta * d
        return vcosts[v] + zcost + xcost

    rrts = rrt.RRTStar(
        np.zeros_like(X),
        400,
        r_rewire=4.0,
        costfn=cost3d,
    )
    xstart = np.array([1, 1])
    xgoal = np.array([35, 25])

    T, gv = rrts.plan(xstart, xgoal)

    fig = go.Figure()

    # get each edge cost
    costs = []
    for e1, e2, data in T.edges(data=True):
        costs.append(data["cost"])
    # normalize edge costs
    norm = colors.Normalize(min(costs), max(costs))
    cmap = cm.get_cmap("coolwarm")

    for (e1, e2), cost in zip(T.edges(), costs):
        color = get_colorstr(cmap(norm(cost)))
        p1, p2 = T.nodes[e1]["pt"], T.nodes[e2]["pt"]
        # plot smooth line
        xys = np.linspace(p1, p2, num=20, axis=0)
        xyzs = np.empty((xys.shape[0], 3))
        for i, xy in enumerate(xys):
            xyzs[i, 0] = xy[0]
            xyzs[i, 1] = xy[1]
            xyzs[i, 2] = gridinterp(xy)
        plots.line_plotly(fig, xyzs, color=color)

    plots.surface_plotly(fig, X, Y, Z, S)
    pyo.plot(fig)
