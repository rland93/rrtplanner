import numpy as np
import networkx as nx
import cvxpy as cp
from rrtplanner import plots


def split_tiles(X, Y, H, nx, ny):
    tiles = {}
    if X.shape[0] % nx != 0:
        raise ValueError("X must be divisible by nx!")
    elif Y.shape[1] % ny != 0:
        raise ValueError("Y must be divisible by ny!")
    for xi in range(nx):
        for yi in range(ny):
            i0 = (X.shape[0] // nx) * xi
            i1 = (X.shape[0] // nx) * (xi + 1)
            j0 = (Y.shape[1] // ny) * yi
            j1 = (Y.shape[1] // ny) * (yi + 1)
            tiles[xi, yi] = {
                "X": X[i0:i1, j0:j1],
                "Y": Y[i0:i1, j0:j1],
                "H": H[i0:i1, j0:j1],
            }
    return tiles


def get_neighbor_tiles(tiles, t):
    i, j = t
    tmaxi = max([t[0] for t in tiles])
    tmaxj = max([t[1] for t in tiles])
    left, right, upper, lower = None, None, None, None
    if i != tmaxi:
        right = i + 1, j
    if i != 0:
        left = i - 1, j
    if j != tmaxj:
        lower = i, j + 1
    if j != 0:
        upper = i, j - 1
    return left, right, upper, lower


def make_tile_edges(tiles):
    for t in tiles:
        for n in get_neighbor_tiles(tiles, t):
            if n is not None:
                yield t, n


def tile_graph(tiles):

    # T is a graph of neighbor relationships between tiles
    T = nx.DiGraph()
    # Add Tstart to the
    for tile, attrs in tiles.items():
        T.add_node(tile, solved=False, **attrs)

    for t, n in make_tile_edges(tiles):
        T.add_edge(t, n)

    # for node, data in T.nodes(data=True):
    #     print(node)
    #     for attr in data:
    #         print(f"\t{attr}")

    return T


def solve_tiles(T, params):
    # Starting tile is the one with the most "terrain variability"
    tstart = max([t for t in T], key=lambda t: np.ptp(T.nodes[t]["H"]))
    # we solve constraint problem
    for i, node in enumerate(nx.bfs_tree(T, tstart)):
        T.nodes[node]["solved_order"] = i
        T.nodes[node]["solved"] = True
        T.nodes[node]["constr"] = []
        # new cvxpy problem
        # terrain height
        S = cp.Variable(T.nodes[node]["H"].shape, name="S")
        constr = []

        # gap safety constraint
        c_gaph = S - T.nodes[node]["H"] >= params["gaph"]
        constr += [c_gaph]

        # calculate derivatives
        dx, dy = cpdiff(S, T.nodes[node]["X"][0, 1] - T.nodes[node]["X"][0, 0], 1)
        d2x, d2y = cpdiff(S, T.nodes[node]["X"][0, 1] - T.nodes[node]["X"][0, 0], 2)

        cost1, cost2, cost3 = 0.0, 0.0, 0.0
        # cost1 += cp.sum_squares(d2x) + cp.sum_squares(d2y)
        cost1 += cp.abs(0.0)
        cost2 += cp.sum(S)
        cost3 += cp.abs(0.0)

        # height constraints
        constr += [cp.max(dx) <= params["maxdx"]]
        constr += [cp.max(dy) <= params["maxdx"]]
        constr += [cp.max(d2x) <= params["maxd2x"]]
        constr += [cp.max(d2y) <= params["maxd2x"]]

        # neighbor constraints
        print(f"{node}: {str(i).rjust(2, ' ')} of {len(T)}")
        for neigh in T.successors(node):
            if T.nodes[neigh]["solved"]:
                # add constraints from neighboring tiles
                node_i, node_j = node
                neig_i, neig_j = neigh
                orient = int(node_i - neig_i), int(node_j - neig_j)
                neighS = T.nodes[neigh]["S"]
                # up
                if orient == (0, -1):
                    print(f"\tUP: {neigh}->{node}")
                    T.nodes[node]["constr"] += ["a"]

                    diff1 = seam_diff(
                        neighS,
                        S,
                        h=params["maxdx"],
                        dir=-1,
                    )

                    diff2 = seam_diff(
                        neighS,
                        S,
                        h=params["maxd2x"],
                        order=2,
                        dir=-1,
                    )

                    diff = cp.sum_squares(diff1) + cp.sum_squares(diff2)

                # down
                elif orient == (0, 1):
                    print(f"\tDOWN: {neigh}->{node}")
                    T.nodes[node]["constr"] += ["b"]

                    diff1 = seam_diff(
                        neighS,
                        S,
                        h=params["maxdx"],
                        dir=1,
                    )

                    diff2 = seam_diff(
                        neighS,
                        S,
                        h=params["maxd2x"],
                        order=2,
                        dir=1,
                    )

                    diff = cp.sum_squares(diff1) + cp.sum_squares(diff2)

                if orient == (1, 0):
                    print(f"\tLEFT: {neigh}->{node}")
                    T.nodes[node]["constr"] += ["l"]
                    diff1 = seam_diff(
                        neighS.T,
                        S.T,
                        h=params["maxdx"],
                        dir=1,
                    )

                    diff2 = seam_diff(
                        neighS.T,
                        S.T,
                        h=params["maxd2x"],
                        order=2,
                        dir=1,
                    )

                    diff = cp.sum_squares(diff1) + cp.sum_squares(diff2)

                elif orient == (-1, 0):
                    print(f"\tRIGHT: {neigh}->{node}")
                    T.nodes[node]["constr"] += ["r"]

                    diff1 = seam_diff(
                        neighS.T,
                        S.T,
                        h=params["maxdx"],
                        dir=-1,
                    )

                    diff2 = seam_diff(
                        neighS.T,
                        S.T,
                        h=params["maxd2x"],
                        order=2,
                        dir=-1,
                    )
                    diff = cp.sum_squares(diff1) + cp.sum_squares(diff2)

                else:
                    diff = None
                if diff is not None:
                    cost3 += diff * 1e4

        problem = cp.Problem(
            objective=cp.Minimize(cost1 + cost2 + cost3), constraints=constr
        )
        problem.solve(solver="GUROBI")
        T.nodes[node]["S"] = S.value
        if problem.status not in ["infeasible", "unbounded"]:
            print("\tfound solution")
            print(f"\t\tcost1: {cost1.value}")
            print(f"\t\tcost2: {cost2.value}")
            print(f"\t\tcost3: {cost3.value}")
        else:
            print("\tinfeasible")
            print(problem.status)
            raise Exception()

    # BFS through tile graph and solve

    # plot the tiles
    # pos = {}
    # for node in T.nodes:
    #     pos[node] = node

    # fig, ax = plt.subplots()
    # nx.draw_networkx(T, pos, arrows=True, ax=ax)
    # plt.show()


def seam_diff(neigh, this, h: float, order: int = 1, dir=1):
    # neighbor to left
    if dir == 1:
        xbak = neigh[:, -1]
        xmid = this[:, 0]
        xfor = this[:, 1]
    # neighbor to right
    elif dir == -1:
        xbak = neigh[:, 0]
        xmid = this[:, -1]
        xfor = this[:, -2]

    if order == 1:
        """first derivative"""
        dx = (xfor - xbak) / (2 * h)
        return dx
    elif order == 2:
        """second derivative"""
        d2x = (xfor - 2 * xmid + xbak) / (h**2)
        return d2x


def cpdiff(x: cp.Variable, h: float, k: int = 1):
    xbak = x[:, 2:]
    xfor = x[:, :-2]
    xmid = x[:, 1:-1]
    ybak = x[2:, :]
    yfor = x[:-2, :]
    ymid = x[1:-1, :]
    if k == 1:
        """first derivative"""
        dx = (xfor - xbak) / (2 * h)
        dy = (yfor - ybak) / (2 * h)
        return dx, dy
    elif k == 2:
        """second derivative"""
        d2x = (xfor - 2 * xmid + xbak) / (h**2)
        d2y = (yfor - 2 * ymid + ybak) / (h**2)
        return d2x, d2y
    else:
        raise ValueError(f"k must be 1 or 2. k={k}")


if __name__ == "__main__":
    from rrtplanner.oggen import example_terrain
    from rrtplanner.plots import plot_surface
    import matplotlib.pyplot as plt

    cols, rows = 80, 80
    n_x, n_y = 4, 4

    X, Y, H = example_terrain(xmax=cols, ymax=rows, hmax=10, cols=cols, rows=rows)

    tiles = split_tiles(X, Y, H, n_x, n_y)

    # fig1, ax1 = plt.subplots()
    # ax1.contour(X, Y, H, levels=10, cmap="gist_earth")

    T = tile_graph(tiles)

    params = {
        "gaph": 5.0,
        "maxdx": 0.2,
        "maxd2x": 0.001,
    }
    solve_tiles(T, params)
    # glue them together
    S = np.empty_like(H)
    centers = {}
    for i, j in np.ndindex(n_x, n_y):
        sizex, sizey = cols // n_x, rows // n_y
        i0, i1 = i * sizex, (i + 1) * sizey
        j0, j1 = j * sizex, (j + 1) * sizey
        S[i0:i1, j0:j1] = T.nodes[(i, j)]["S"]
        centers[(i, j)] = i0 + (i1 - i0) // 2, j0 + (j1 - j0) // 2

    """
    # individual plots
    fig2, ax2 = plt.subplots(nrows=n_y, ncols=n_x, tight_layout=True)
    for i, j in np.ndindex(n_x, n_y):
        ax2[i, j].contour(
            T.nodes[(i, j)]["X"],
            T.nodes[(i, j)]["Y"],
            T.nodes[(i, j)]["H"],
            levels=5,
            cmap="gist_earth",
        )
        # ax2[i, j].set_title(f"({i}, {j})")

    fig3, ax3 = plt.subplots(nrows=n_y, ncols=n_x, tight_layout=True)
    for i, j in np.ndindex(n_x, n_y):
        ax3[i, j].contour(
            T.nodes[(i, j)]["X"],
            T.nodes[(i, j)]["Y"],
            T.nodes[(i, j)]["S"],
            levels=6,
            cmap="bone",
        )
        # ax2[i, j].set_title(f"({i}, {j})")

    fig4 = plt.figure()
    ax4 = fig4.add_subplot(projection="3d")
    plots.plot_3D_terrain(
        ax4, X, Y, H, zsquash=0.25, wireframe=False, cmap="gist_earth"
    )
    plots.plot_3D_terrain(ax4, X, Y, S, zsquash=0.25, wireframe=True, cmap="bone")
    """

    fig5, (ax51, ax52, ax53) = plt.subplots(ncols=3)
    ax51.contour(X, Y, H, levels=10, cmap="gist_earth")
    ax52.imshow(S, cmap="coolwarm", origin="lower")
    ax53.imshow(np.gradient(S), cmap="coolwarm", origin="lower")
    for node in T.nodes:
        x, y = centers[node]
        text = str(T.nodes[node]["solved_order"])
        text += "\n"
        text += " ".join([str(c) for c in T.nodes[node]["constr"]])
        ax52.text(x, y, text)
        ax53.text(x, y, text)

    for a in (ax51, ax52, ax53):
        a.set_aspect("equal")

    # fig6, ax6 = plt.subplots()
    # ax6.imshow(S, cmap="viridis", origin="lower")

    plt.show()
