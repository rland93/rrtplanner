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


def get_path_3d(T, X, Y, S, nodes, n=10):
    gridinterp = get_grid_interp(X, Y, S)
    lines = []
    for node in nodes[:-1]:
        # unpack points from i, j -> x, y
        p1i = T.nodes[node]["pt"][0]
        p1j = T.nodes[node]["pt"][1]
        p2i = T.nodes[nodes[nodes.index(node) + 1]]["pt"][0]
        p2j = T.nodes[nodes[nodes.index(node) + 1]]["pt"][1]
        p1 = np.array([X[p1i, p1j], Y[p1i, p1j]])
        p2 = np.array([X[p2i, p2j], Y[p2i, p2j]])
        # interpolate (x, y) -> z and create 3-D array
        linex = np.linspace(p1, p2, num=n)
        linez = np.atleast_2d(gridinterp(linex)).T
        line3d = np.concatenate((linex, linez), axis=1)
        lines.append(line3d)
    lines = np.concatenate(lines, axis=0)
    return lines, list()


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
    zcost = 10.0 * s
    xcost = 1.0 * d
    return vcosts[v] + zcost + xcost


if __name__ == "__main__":
    from plotly import offline as pyo
    from plotly import graph_objects as go
    from matplotlib import colors, cm
    from matplotlib import pyplot as plt
    import pathlib, os

    PLOT_DIR = pathlib.Path(__file__).parent.resolve()
    PLOT_DIR = PLOT_DIR / "../../surfaceplanner/urop/"
    PLOT_DIR = (PLOT_DIR / "figures").resolve()
    os.makedirs(PLOT_DIR, exist_ok=True)
    print("Saving plots to ", PLOT_DIR)

    PLOT_METH = "plotly"

    PROBLEMS = {
        "Low": True,
        "Drum": False,
        "Waypoints": False,
        "3D": False,
    }

    TERRAIN = {
        "M": 60,
        "N": 60,
        "hmax": 25,
        "xmax": 60,
        "ymax": 60,
    }
    CONSTRAINTS = {
        "gaph": 4.0,
        "minh": 2.5,
        "maxdx": 0.9,
        "maxd2x": 0.08,
    }
    X, Y, H = oggen.example_terrain(
        TERRAIN["xmax"],
        TERRAIN["ymax"],
        TERRAIN["hmax"],
        TERRAIN["M"],
        TERRAIN["N"],
    )

    if PROBLEMS["Low"]:
        ### create and solve the optimization problem
        lowsurf = surface.SurfaceLowHeight(X, Y, H)
        lowsurf.setup(
            CONSTRAINTS["minh"],
            CONSTRAINTS["gaph"],
            CONSTRAINTS["maxdx"],
            CONSTRAINTS["maxd2x"],
            use_parameters=False,
        )
        lowsurf.solve(verbose=True, solver="GUROBI")
        for objective, value in lowsurf.objectives.items():
            print(f"{objective} : {value}")
        S = lowsurf.S.value
        ### Get an occupancy Grid
        rrts = rrt.RRTStar(
            og=oggen.corridor_og(X.shape),
            n=250,
            r_rewire=TERRAIN["M"] / 2.0,
            costfn=cost3d,
        )
        sx, sy = 1, 3 * Y.shape[1] // 4
        xstart = np.array([sx, sy])
        gx, gy = int(X.shape[0] - 1), Y.shape[1] // 4
        xgoal = np.array([gx, gy])
        T, gv = rrts.plan(xstart, xgoal)
        lines, costs = get_layout_3d(T, X, Y, S, n=20)

        path = rrts.route2gv(T, gv)

        # get lines
        pathline, _ = get_path_3d(T, X, Y, S, path, n=20)

        if PLOT_METH == "plotly":
            fig = go.Figure()
            plots.put_lines_plotly(fig, lines, costs, cmap="coolwarm", alpha=0.2)
            plots.surface_plotly(fig, X, Y, H, S)
            plots.line_plotly(fig, pathline, color="red", width=10)
            pyo.plot(fig, filename=str(PLOT_DIR / "low_height.html"))
        elif PLOT_METH == "matplotlib":
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            plots.plot_surface(ax, X, Y, S, zsquash=1.0, cmap="bone")
            plots.plot_surface(
                ax, X, Y, H, zsquash=1.0, cmap="gist_earth", wireframe=False
            )
            plt.show()

    if PROBLEMS["Waypoints"]:

        waypoints = surface.generate_example_waypoints(
            X,
            Y,
            H,
            nwaypoints=3,
            above_range=(5, 10),
        )

    if PROBLEMS["3D"]:
        # show "corridor"
        og = oggen.corridor_og(X.shape)
        rrts = surface.RRT3D(og, 250, r_rewire=30.0, ogz=H.T, heightband=10.0, gap=5.0)
        sx, sy = 1, 3 * Y.shape[1] // 4
        xstart = np.array([sx, sy, H[sx, sy] + 5.0])
        gx, gy = int(X.shape[0] - 1), Y.shape[1] // 4
        xgoal = np.array([gx, gy, H[gx, gy] + 5.0])
        T, gv = rrts.plan(xstart, xgoal)
        path = rrts.route2gv(T, gv)

        # get lines
        pathline = []
        for node in path:
            print(node, T.nodes[node]["pt"], gv)
            pathline.append(T.nodes[node]["pt"])
        pathline = np.array(pathline)

        # create figure
        fig = go.Figure()

        # get each edge cost
        costs = []
        for e1, e2, data in T.edges(data=True):
            costs.append(data["cost"])
        # normalize edge costs
        norm = colors.Normalize(min(costs), max(costs))
        cmap = cm.get_cmap("coolwarm")

        for (e1, e2), cost in zip(T.edges(), costs):
            color = plots.get_colorstr(cmap(norm(cost)))
            p1, p2 = T.nodes[e1]["pt"], T.nodes[e2]["pt"]
            # plot smooth line
            xyzs = np.linspace(p1, p2, num=2, axis=0)
            plots.line_plotly(fig, xyzs, color=color)

        plots.line_plotly(fig, pathline, color="red", width=10)
        plots.surface_plotly(fig, X, Y, H, S=None)
        pyo.plot(fig, filename=str(PLOT_DIR / "3D.html"))
