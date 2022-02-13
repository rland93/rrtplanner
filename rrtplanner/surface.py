from rrt import RRT
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import interpolate
import cvxpy as cp


class RRTSurface(RRT):
    def __init__(self, og: np.ndarray, n: int, H: np.ndarray):
        if og.shape != H.shape:
            raise ValueError("og and H must have the same shape")
        self.og = og
        self.H = H
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
        points = np.full((self.n, 2), dtype=int, fill_value=self.not_a_point)
        vcosts = np.full((self.n,), fill_value=self.not_a_dist)
        children, parents = defaultdict(list), {}
        points[0] = xstart
        vcosts[0] = 0
        parents[0] = None
        i, j = 0, 1
        if self.pbar:
            pbar = tqdm(total=self.n)

        while i < self.n:
            if self.pbar:
                pbar.update(1)

            xnew = self.sample_all_free()
            vnearest = self.near(points, xnew)[0]
            xnearest = points[vnearest]

            nocoll = self.collisionfree(self.og, xnearest, xnew)
            if nocoll and tuple(xnew) not in sampled and j != self.n:
                sampled.add(tuple(xnew))

                # check least cost path to xnew
                vbest = vnearest
                cbest = self.cost(vcosts, points, vbest, xnew)
                vnear = self.within(points, xnew, self.r_rewire)

                for vn in vnear:
                    xn = points[vn]
                    cn = self.cost(vcosts, points, vn, xnew)
                    if cn < cbest:
                        if self.collisionfree(self.og, xn, xnew):
                            vbest = vn
                            cbest = cn

                # store new point
                vnew = j
                points[vnew] = xnew
                vcosts[vnew] = cbest
                # store new edge
                parents[vnew] = vbest
                children[vbest].append(vnew)

                # tree rewire
                for vn in vnear:
                    xn = points[vn]
                    cn = vcosts[vn]
                    cmaybe = self.cost(vcosts, points, vn, xnew)
                    if cmaybe < cn:
                        if self.collisionfree(self.og, xn, xnew):
                            parent = parents[vn]
                            if parent is not None:
                                # reassign parent
                                try:
                                    children[parent].remove(vn)
                                    parents[vn] = vnew
                                    vcosts[vn] = cmaybe
                                except ValueError:
                                    pass
                j += 1
            i += 1
        # go to goal if possible
        vgoal, children, parents, points, vcosts = self.go2goal(
            vcosts, points, xgoal, j, children, parents
        )
        # build graph
        T = self.build_graph(vgoal, points, parents, vcosts)

        return T, vgoal


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


def optimal_path(x, h, smin=10.0, sbuf=5.0, ds=1.5, d2s=0.02):
    assert x.shape[0] == h.shape[0]
    x_gap = np.linalg.norm(x[0, :] - x[1, :])
    s = cp.Variable(h.shape)
    cost = cp.sum(s)

    smin_c = s >= smin
    sbuf_c = s - h >= sbuf
    ds_c = cp.abs(cp.diff(s)) - ds * x_gap <= 0
    d2s_c = cp.abs(cp.diff(s, 2)) - d2s * x_gap**2 <= 0

    constraints = [smin_c, sbuf_c, ds_c, d2s_c]
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

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
    print(problem.status)
    if any([problem.status == status for status in viable_status]):
        return s.value
    elif any([problem.status == status for status in non_viable_status]):
        return None


if __name__ == "__main__":
    w, h = 300, 200
    og, xy = make_circle_og(w, h, w / 8)
    og *= 50
    grid_interp = interpolate.RegularGridInterpolator(
        (np.arange(w), np.arange(h)), og, method="linear"
    )

    # three points
    x1 = np.array([10, 10])
    x2 = np.array([270, 180])
    x3 = np.array([165, 180])
    x4 = np.array([190, 180])

    # two lines
    line_n = 100
    xa = np.linspace(x1, x2, line_n)
    xb = np.linspace(x1, x3, line_n)
    xc = np.linspace(x1, x4, line_n)

    ta = np.linalg.norm((x1 - x2)) / line_n * np.arange(line_n)
    tb = np.linalg.norm((x1 - x3)) / line_n * np.arange(line_n)
    tc = np.linalg.norm((x1 - x4)) / line_n * np.arange(line_n)

    ha = grid_interp(xa)
    hb = grid_interp(xb)
    hc = grid_interp(xc)

    sa = optimal_path(xa, ha)
    sb = optimal_path(xb, hb)
    sc = optimal_path(xc, hc)

    fig1, (ax1, ax2) = plt.subplots(nrows=2, tight_layout=True, dpi=200)
    ax1.imshow(og.T, cmap="binary")
    ax1.plot([x1[0], x3[0]], [x1[1], x3[1]], "b:", label="path(a)")
    ax1.plot([x1[0], x4[0]], [x1[1], x4[1]], "r-.", label="path(b)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.set_title("Aerial View of Paths with Obstacle")

    ax2.set_title("Height of Paths")
    ax2.plot(tb, hb, linestyle="-", color="blue", label="terrain height (a)")
    ax2.plot(tc, hc, linestyle="-", color="red", label="terrain height (b)")
    ax2.plot(tb, sb, linestyle=":", color="blue", label="optimal path (a)")
    ax2.plot(tc, sc, linestyle="-.", color="red", label="optimal path (b)")
    ax2.set_xlabel("distance along path")
    ax2.set_ylabel("height")
    ax2.set_aspect("equal")
    ax2.legend()

    plt.show()
