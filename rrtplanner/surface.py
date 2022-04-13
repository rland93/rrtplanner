<<<<<<< HEAD
from rrt import RRT
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import interpolate
import cvxpy as cp
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Line3DCollection


class RRTSurface(RRT):
    def __init__(self, og: np.ndarray, n: int, r_rewire: float, zterrain: np.ndarray):
        if og.shape != zterrain.shape:
            raise ValueError("og and H must have the same shape")
        self.og = og
        self.zterrain = zterrain

        # interpolator for zterrain
        self.zinterp = interpolate.RegularGridInterpolator(
            (np.arange(og.shape[0]), np.arange(og.shape[1])), zterrain, method="linear"
        )

        self.r_rewire = r_rewire

        self.zmin = 1.0
        self.zmax = 8.0
        self.zbuf = 0.5

        super().__init__(og, n)

    def plan(self, xstart: np.ndarray, xgoal: np.ndarray):
        """
        Compute a plan from `xstart` to `xgoal`. Using the RRT* algorithm.

        The plan is a tree, with the root at
        `xstart` and a leaf at `xgoal`. If xgoal could not be connected to the tree, the
        leaf nearest to xgoal is considered the "goal" leaf.

        Parameters
        ----------
        xstart : np.ndarray
            (2, ) start point
        xgoal : np.ndarray
            (2, ) goal point

        Returns
        -------
        Tuple[nx.DiGraph, int]
            DiGraph of the tree, and the vertex of the goal leaf (if goal could be
            reached) or the closest tree node.
        """
        sampled = set()

        # xs is now 3-dimensional
        zs = np.zeros((self.n, 1), dtype=float)
        xs = np.full((self.n, 2), dtype=int, fill_value=self.not_a_point)
        vcosts = np.full((self.n,), fill_value=self.not_a_dist)
        children, parents = defaultdict(list), {}
        # contains z-path from parent to child
        paths = {}

        # x, y are the same
        xs[0] = xstart
        # z buffer + terrain or z min whichever is larger
        zs[0] = max(self.zinterp(xstart) + self.zbuf, self.zmin) + 0.01

        vcosts[0] = 0
        parents[0] = None
        i, j = 0, 1

        if self.pbar:
            pbar = tqdm(total=self.n)

        while i < self.n:
            if self.pbar:
                pbar.update(1)

            # new xy
            xnew = self.sample_all_free()
            vnearest = self.near(xs, xnew)[0]
            # nearest point
            xnearest = xs[vnearest]

            # get new optimal path
            opt_path_x = np.linspace(xnearest, xnew, num=25)
            opt_terrain_z = self.zinterp(opt_path_x)
            z0 = np.squeeze(zs[vnearest])
            opt_path = optimal_path(
                opt_path_x,
                opt_terrain_z,
                z0,
                smin=self.zmin,
                smax=self.zmax,
                sbuf=self.zbuf,
                plot=False,
                og=self.zterrain,
            )
            opt_path_exists = opt_path is not None

            if opt_path_exists and tuple(xnew) not in sampled and j != self.n:
                sampled.add(tuple(xnew))
                vbest = vnearest

                # store new point
                vnew = j
                xs[vnew] = xnew
                zs[vnew] = opt_path[-1]

                # store new edge
                parents[vnew] = vbest
                children[vbest].append(vnew)
                paths[vnew] = {"z": opt_path, "x": opt_path_x, "t": opt_terrain_z}

                j += 1
            i += 1
        # build graph
        T = nx.DiGraph()
        T.add_node(0, pt=xs[0], z=zs[0])
        for c, p in parents.items():
            if p is not None:
                T.add_edge(p, c, path=paths[c], cost=0.0)
                T.add_node(c, pt=xs[c], z=zs[c])
        return T


def plot_edges_3d(ax, T):
    # get normalization
    allz = np.concatenate([T[e1][e2]["path"]["z"] for e1, e2 in T.edges()])
    maxz, minz = allz.max(), allz.min()
    norm = Normalize(vmin=minz, vmax=maxz)

    allxy = np.concatenate([T[e1][e2]["path"]["x"] for e1, e2 in T.edges()], axis=0)
    maxx, minx = allxy[:, 0].max(), allxy[:, 0].min()
    maxy, miny = allxy[:, 1].max(), allxy[:, 1].min()

    for e1, e2 in T.edges():
        path = T[e1][e2]["path"]
        lines = []
        for i in range(len(path["z"]) - 1):
            # get points
            p1 = np.empty((3,))
            p2 = np.empty((3,))

            p1[1] = path["x"][i][0]
            p2[1] = path["x"][i + 1][0]
            p1[0] = path["x"][i][1]
            p2[0] = path["x"][i + 1][1]
            p1[2] = path["z"][i]
            p2[2] = path["z"][i + 1]
            lines.append((p1, p2))

        l3d = Line3DCollection(
            lines,
            color=cm.viridis(norm(path["z"])),
        )
        ax.add_collection(l3d)
        ax.set_ylim3d(minx, maxx)
        ax.set_xlim3d(miny, maxy)
        ax.set_zlim3d(minz, maxz)


def make_circle_og(w, h, r):
    """Make a grid containing a circle with radius R
    where the points inside the circle are 1 and the points outside the circle are 0

    Parameters
    ----------
    wh : tuple
        width and height of the grid
    r : int
        radius of the circle
    """
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    xy = np.stack((x, y), axis=2)
    grid = np.where(
        (xy[:, :, 0] - w / 2) ** 2 + (xy[:, :, 1] - h / 2) ** 2 < r**2, 1, 0
    )
    return grid.T, xy


def optimal_path(
    x,
    z_terrain,
    z0,
    smin=10.0,
    smax=45.0,
    sbuf=5.0,
    ds=2.5,
    d2s=0.06,
    plot=False,
    og=None,
):
    if x.shape[0] != z_terrain.shape[0]:
        raise ValueError(
            "x and z_terrain must have same no. of points. x.shape={}, z_terrain.shape={}".format(
                x.shape, z_terrain.shape
            )
        )
    # gap between regularly spaced points
    x_gap = np.linalg.norm(x[0, :] - x[1, :])

    # surface variable
    s = cp.Variable(z_terrain.shape, name="s")
    # first point constraint
    s0_c = s[0] == z0
    # min height constraints
    smin_c = s >= smin
    smax_c = s <= smax
    # buffer constraints
    sbuf_c = s - z_terrain >= sbuf
    # diff constraints
    ds_c = cp.abs(cp.diff(s)) - ds * x_gap <= 0
    d2s_c = cp.abs(cp.diff(s, 2)) - d2s * x_gap**2 <= 0

    # add all constraints
    constraints = [s0_c, sbuf_c, smin_c, smax_c, d2s_c, d2s_c]

    cost = cp.sum(s)

    # cost: stay low as possible
    problem = cp.Problem(cp.Minimize(cost), constraints)

    # solve problem
    problem.solve()

    # non viable solutions return none
    viable_status = [
        cp.OPTIMAL,
        cp.OPTIMAL_INACCURATE,
    ]
    non_viable_status = [
        cp.INFEASIBLE,
        cp.UNBOUNDED,
        cp.INFEASIBLE_INACCURATE,
        cp.UNBOUNDED_INACCURATE,
    ]
    if plot:
        print(problem.status)

        fig, (ax1, ax2) = plt.subplots(ncols=2)
        if any([problem.status == status for status in viable_status]):
            ax1.plot(np.arange(z_terrain.shape[0]), s.value, "r--")

        ax1.plot(np.arange(z_terrain.shape[0]), z_terrain + sbuf, "g:")
        ax1.plot(np.arange(z_terrain.shape[0]), np.full(z_terrain.shape, smin), "m:")

        ax1.plot(np.arange(z_terrain.shape[0]), z_terrain, "k")
        ax1.scatter(0, z0, c="r")
        ax2.imshow(og.T, cmap="binary", origin="lower")
        ax2.plot(x[:, 0], x[:, 1], c="k")
        plt.show()
    # get value if problem was solved
    if any([problem.status == status for status in viable_status]):
        return s.value
    # otherwise, return None
    elif any([problem.status == status for status in non_viable_status]):
        return None


if __name__ == "__main__":
    from mayavi import mlab
    from oggen import perlin_occupancygrid
    import rrt, plots

    w, h = 300, 200
    X, Y = np.meshgrid(np.arange(h), np.arange(w))
    zterrain = perlin_occupancygrid(w, h, 0.3) * 3
    zterrain |= perlin_occupancygrid(w, h, 0.3) * 5
    og = np.zeros_like(zterrain)

    rrts = RRTSurface(og, 2000, 50, zterrain)
    xstart = rrts.sample_all_free()
    xgoal = rrts.sample_all_free()
    T = rrts.plan(xstart, xgoal)

    # plot tree
    for e1, e2, d in T.edges(data=True):
        z = T[e1][e2]["path"]["z"]
        x = T[e1][e2]["path"]["x"][:, 0]
        y = T[e1][e2]["path"]["x"][:, 1]
        mlab.plot3d(y, x, z, tube_radius=0.1, color=(0, 0.2, 1.0))

    # plot surface
    mlab.mesh(X, Y, zterrain, colormap="gist_earth")

    mlab.show()
    """

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")

    plots.plot_rrt_lines(ax1, T)
    plots.plot_og(ax1, zterrain)
    plots.plot_3D_terrain(ax2, X, Y, zterrain, zsquash=0.1, alpha=1.0)

    plot_edges_3d(ax3, T)
    # plots.plot_3D_terrain(ax3, X, Y, zterrain, zsquash=0.2, alpha=0.3)
    plt.show()
    """
=======
import cvxpy as cp
import numpy as np


def getsurface(
    X, Y, zterrain, zgap, zmin, xystep, vmin, amin, waypoints=None, waypointcost=None
):
    """Get grid from occupancygrid og"""
    S = cp.Variable(shape=zterrain.shape)

    cost, constraints = 0.0, []
    # constraints
    constraints.append(S >= zmin)  # min height
    constraints.append(S - zterrain >= zgap)  # gap

    # first deriv constraints
    dx = cp.abs(cp.diff(S, axis=0))
    dy = cp.abs(cp.diff(S, axis=1))
    dxc = dx <= vmin * xystep
    dyc = dy <= vmin * xystep
    constraints.append(dxc)
    constraints.append(dyc)

    # second deriv constraints
    ddx = cp.abs(cp.diff(S, axis=0, k=2))
    ddy = cp.abs(cp.diff(S, axis=1, k=2))
    ddxc = ddx <= amin * xystep**2
    ddyc = ddy <= amin * xystep**2
    constraints.append(ddxc)
    constraints.append(ddyc)

    # waypoint constraints
    if waypoints is not None:
        for wp in waypoints:
            # get quadrilateral
            wpy_i = np.argsort(np.abs(X - wp[0]), axis=1)[0][:4]
            wpx_i = np.argsort(np.abs(Y - wp[1]), axis=0)[:4][:, 0]
            # average of quadrilateral
            cost += cp.sum_squares(S[wpx_i, wpy_i] - wp[2]) * waypointcost / 4.0

    # per-grid point cost
    cost += cp.sum(S) / (S.shape[0] * S.shape[1])

    # solve
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver="ECOS", verbose=True)

    # log waypoint error
    if waypoints is not None:
        for wp in waypoints:
            i = np.argsort(np.abs(X - wp[0]), axis=1)[0][:4]
            j = np.argsort(np.abs(Y - wp[1]), axis=0)[:4][:, 0]
            wp_error = np.mean(np.abs(S.value[i, j] - wp[2]))
            print(wp_error)

    # get surface
    return S.value


if __name__ == "__main__":
    import os
    from oggen import perlin_terrain
    from plots import plot_3D_terrain
    from random import uniform
    from matplotlib import pyplot as plt

    # z squash
    ZSQ = 0.33

    # terrain
    tcmap = "gist_earth"

    figsz = (6, 6)

    w, h = 160, 160
    ridgemid = w / 2.0
    ridgewid = w / 5.0
    step = 1.0
    z_gap = 15.0
    z_min = 25.0

    dx = 1.4
    ddx = 0.10

    nwaypoints = 2

    # get some random terrain
    alpha = 80  # high freq scale
    beta = 50  # low freq scale
    X, Y = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
    Z = perlin_terrain(w, h, scale=1.5) * alpha + perlin_terrain(w, h, scale=0.3) * beta

    ridge = np.empty((w, h))
    ridgex = X[0, :]

    def ridge_fn(x, mid, width):
        return np.exp(-(((x - mid) / width) ** 2))

    ridgey = ridge_fn(ridgex, ridgemid, ridgewid)
    for i in range(0, w):
        ridge[i, :] = ridgey

    Z *= ridge

    terrain_fig_test = plt.figure()
    ax = terrain_fig_test.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap=tcmap)
    plt.show()

    fig0 = plt.figure(tight_layout=True, figsize=figsz)
    ax0 = fig0.add_subplot(111, projection="3d")
    ax0 = plot_3D_terrain(ax0, X, Y, Z, zsquash=ZSQ, cmap=tcmap)
    ax0 = plot_3D_terrain(
        ax0, X, Y, Z + z_gap, zsquash=ZSQ, wireframe=True, cmap="bone"
    )
    ax0.set_title(r"Naive: $z_{sheet} = z_{terrain} + z_{gap}$")

    # 1st Derivative Only
    S1 = getsurface(X, Y, Z, z_gap, z_min, step, dx, 1000.0)
    fig1 = plt.figure(tight_layout=True, figsize=figsz)
    ax1 = fig1.add_subplot(111, projection="3d")
    ax1 = plot_3D_terrain(ax1, X, Y, Z, zsquash=ZSQ, cmap=tcmap)
    ax1 = plot_3D_terrain(ax1, X, Y, S1, zsquash=ZSQ, wireframe=True, cmap="bone")
    ax1.set_title(r"Optimal: Gap and 1st Deriv")

    # 2nd Derivative + 1st Derivative
    S2 = getsurface(X, Y, Z, z_gap, z_min, step, dx, ddx)
    fig2 = plt.figure(tight_layout=True, figsize=figsz)
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2 = plot_3D_terrain(ax2, X, Y, Z, zsquash=ZSQ, cmap=tcmap)
    ax2 = plot_3D_terrain(ax2, X, Y, S2, zsquash=ZSQ, wireframe=True, cmap="bone")
    ax2.set_title(r"Optimal: Gap, 1st, 2nd Deriv Constraints")

    # 1st, 2nd, and Waypoints
    fig3 = plt.figure(tight_layout=True, figsize=figsz)
    ax3 = fig3.add_subplot(111, projection="3d")

    waypoints = []
    waypoints.append([6 * w // 7, 6 * h // 7, max(alpha, beta) + 40.0])
    waypoints.append([w // 7, h // 7, max(alpha, beta) + 25.0])
    waypoints = np.array(waypoints)

    S3 = getsurface(
        X, Y, Z, z_gap, z_min, step, dx, ddx, waypoints=waypoints, waypointcost=1e3
    )

    ax3 = plot_3D_terrain(ax3, X, Y, Z, zsquash=ZSQ, cmap=tcmap)
    ax3 = plot_3D_terrain(ax3, X, Y, S3, zsquash=ZSQ, wireframe=True, cmap="bone")
    ax3.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], c="r", s=40)
    ax3.set_title("Optimal: Gap, 1st, 2nd Deriv Constraints, Waypoint Costs")

    for a in [ax0, ax1, ax2, ax3]:
        a.set_xticks([])
        a.set_yticks([])
        a.set_zticks([])

    plt.show()
    if not os.path.exists("./figure_output"):
        os.makedirs("./figure_output")
    fig0.savefig("./figure_output/scratch.png", dpi=300)
    for i, f in enumerate([fig0, fig1, fig2, fig3]):
        print("Saving figure {}/4".format(i))
        f.savefig("./figure_output/{}.png".format(i), dpi=300)
>>>>>>> dev
