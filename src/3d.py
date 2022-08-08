from rrtplanner import plots, surface, rrt, oggen
import numpy as np


if __name__ == "__main__":
    from plotly import graph_objects as go
    from plotly import offline as pyo
    from matplotlib import colors, cm

    W, H, D = 100, 100, 25
    xmax, ymax = W, H
    cols, rows = W, H
    X, Y, Z = oggen.example_terrain(xmax=xmax, ymax=ymax, cols=cols, rows=rows)
    Z *= float(D) / Z.ptp()

    # make a drum
    Z = np.zeros_like(X)
    for (i, j) in np.ndindex(Z.shape):
        if rrt.r2euc([X[i, j], Y[i, j]], [W / 2, H / 2]) < W / 5.0:
            Z[i, j] = 40.0
        else:
            Z[i, j] = 0.0

    rrts = surface.RRT3D(
        np.zeros_like(X),
        500,
        r_rewire=40.0,
        ogz=Z.T,
        heightband=40.0,
    )
    xstart = np.array([1, 1, Z[1, 1] + 60.0])
    xgoal = np.array([35, 25, Z[35, 25] + 60.0])
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
        color = plots.get_colorstr(cmap(norm(cost)))
        p1, p2 = T.nodes[e1]["pt"], T.nodes[e2]["pt"]
        # plot smooth line
        xyzs = np.linspace(p1, p2, num=2, axis=0)
        plots.line_plotly(fig, xyzs, color=color)

    plots.surface_plotly(fig, X, Y, Z, S=None)
    pyo.plot(fig)
