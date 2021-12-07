import numpy as np
import networkx as nx
from scipy.linalg import svd, det, norm
from tqdm import tqdm
from matplotlib.patches import Ellipse

# from matplotlib import cm
# from matplotlib.colors import Normalize
# import matplotlib.pyplot as plt
from math import sqrt, sin, cos
import numba


def get_rand_start_end(world, bias=True):
    """get random free start, end in the world"""
    free = np.argwhere(world == 0)
    """if bias, prefer points far away from one another"""
    if bias == True:
        start_i = int(np.random.beta(a=0.5, b=5) * free.shape[0])
        end_i = int(np.random.beta(a=5, b=0.5) * free.shape[0])
    else:
        start_i = np.random.choice(free.shape[0])
        end_i = np.random.choice(free.shape[0])
    start = free[start_i, :]
    end = free[end_i, :]
    return start, end


@numba.njit(fastmath=True)
def sample_all_free(free):
    """sample uniformly from free space in the world."""
    return free[np.random.choice(free.shape[0])]


@numba.njit(fastmath=True)
def norm(u, v):
    d = u - v
    return sqrt(d[0] * d[0] + d[1] * d[1])


@numba.njit(fastmath=True)
def collisionfree(world, a, b) -> bool:
    """calculate linear collision on world between points a, b"""
    x0 = a[0]
    y0 = a[1]
    x1 = b[0]
    y1 = b[1]
    dx = abs(x1 - x0)
    if x0 < x1:
        sx = 1
    else:
        sx = -1
    dy = -abs(y1 - y0)
    if y0 < y1:
        sy = 1
    else:
        sy = -1
    err = dx + dy
    x0_old, y0_old = x0, y0
    while True:
        if world[x0, y0] == 1:
            return False, np.array([x0_old, y0_old])
        elif x0 == x1 and y0 == y1:
            return True, None
        else:
            x0_old, y0_old = x0, y0
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy


def nearest(points, x):
    """get a list of the nodes sorted by distance to point `x`, nearest first"""
    dists = np.linalg.norm((points - x), axis=1)
    dists_sorted = np.argsort(dists)
    return dists_sorted


class RRTvec(object):
    """base class containing common RRT methods"""

    def __init__(self, world, n, dubins=False, pbar=True):
        # the world
        self.world = np.array(world)
        # whether to display a progress bar
        self.pbar = pbar
        # array containing vertex points
        self.n = n
        # points, N+1 x 2
        self.points = np.full((n, 2), np.iinfo(np.int64).max, dtype=np.int64)
        # edges (matrix) N+1 x N+1
        self.edges = np.zeros((n, n), dtype=bool)
        # costs (array) N+1 x 1
        self.vcosts = np.full((n, 1), np.Infinity, dtype=float)
        # world free space
        self.free = np.argwhere(world == 0)

    def clamp_arr(self, point):
        """clamp input 2d array so that it is in the world"""
        wshape = self.world.shape
        # clamp lower
        c_lower = np.maximum(np.array([0, 0]), point)
        # clamp upper
        c_upper = np.minimum(np.array(wshape), c_lower)
        return np.int64(c_upper)

    def get_parent(self, v):
        """get first parent of node `v`. If a parent does not exist,
        return None."""
        return np.argwhere(self.edges[:, v] == 1)[0]

    def near(self, x, r):
        """get nodes within `r` of point `x`"""
        dists = np.linalg.norm(self.points - x, axis=1)
        return np.argwhere(dists < r)

    def update_world(self, neww):
        self.world = neww
        self.free = np.argwhere(neww == 0)

    def calc_cost(self, v, x):
        return self.vcosts[v] + norm(self.points[v], x)

    def try_goal(self, xgoal):
        self.points[-1] = xgoal
        dists = np.linalg.norm(self.points - xgoal)
        closest = np.argsort(dists)
        dists = dists[closest]
        idxs = np.arange(0, self.n + 2)[closest]
        points = self.points[closest]
        for i, p, d in zip(idxs, points, dists):
            if collisionfree(self.world, p, xgoal):
                self.edges[i, -1] = 1


class RRTstar(RRTvec):
    def __init__(self, world, n):
        super().__init__(world, n)

    def make(self, xstart, xgoal, r_rewire):
        xstart = self.clamp_arr(xstart)
        xgoal = self.clamp_arr(xgoal)

        # store xstart into points
        self.points[0, :] = xstart
        # store costs
        self.vcosts[0] = 0.0
        i = 1
        if self.pbar:
            pbar = tqdm(total=self.n, desc="tree")
        while i < self.n:
            xnew = sample_all_free(self.free)
            vnearest = nearest(self.points, xnew)
            xnearest = self.points[vnearest[0]]
            nocoll = collisionfree(self.world, xnearest, xnew)

            if nocoll[0] and np.all(xnearest != xnew):
                vnew = i
                vbest = vnearest[0]

                vnear = self.near(xnew, r_rewire)
                vnear = vnear.reshape(vnear.shape[0])

                cbest = self.calc_cost(vbest, xnew)
                for vn in vnear:
                    xn = self.points[vn]
                    cn = self.calc_cost(vn, xnew)
                    if cn < cbest:
                        if collisionfree(self.world, xn, xnew)[0]:
                            cbest = cn
                            vbest = vn

                # store new point
                self.points[i] = xnew
                self.vcosts[i] = cbest
                # store new edge
                self.edges[vbest, vnew] = 1

                # tree rewire
                for vn in vnear:
                    xn = self.points[vn]
                    cn = self.vcosts[vn]
                    possible_cost = self.calc_cost(vn, xnew)
                    if possible_cost < cn:
                        if collisionfree(self.world, xn, xnew)[0]:
                            parent = self.get_parent(vn)
                            if parent is not None:
                                self.edges[parent, vn] = 0
                                self.edges[vnew, vn] = 1
                                self.vcosts[vn] = possible_cost
                if self.pbar:
                    pbar.update(1)
                i += 1

        T = nx.convert_matrix.from_numpy_array(self.edges, create_using=nx.DiGraph)
        for e1, e2 in T.edges:
            T[e1][e2]["cost"] = self.vcosts[e1]

        if self.pbar:
            pbar.update(1)
            pbar.close()

        return T


class RRTstandard(RRTvec):
    def __init__(self, world, n):
        super().__init__(world, n)

    def make(self, xstart, xgoal):
        """Make RRT standard tree with `N` points from xstart to xgoal.
        Returns the tree, the start node, and the end node."""
        xstart = self.clamp_arr(xstart)
        xgoal = self.clamp_arr(xgoal)
        # store xstart into points
        self.points[0, :] = xstart
        # store costs
        self.vcosts[0] = 0.0
        i = 1
        if self.pbar:
            pbar = tqdm(total=self.n, desc="tree")
        while i < self.n:
            xnew = sample_all_free(self.free)
            vnearest = self.nearest_nodes(xnew)
            xnearest = self.points[vnearest[0]]
            nocoll = collisionfree(self.world, xnearest, xnew)
            if nocoll[0] and np.all(xnearest != xnew):
                vnew = i
                # store point
                self.points[vnew] = xnew
                # store cost
                self.vcosts[vnew] = self.calc_cost(vnearest[0], x=xnew)
                # store edge
                self.edges[vnearest[0], vnew] = 1

                if self.pbar:
                    pbar.update(1)
                i += 1
        if self.pbar:
            pbar.update(1)
            pbar.close()

        T = nx.convert_matrix.from_numpy_array(self.edges.T, create_using=nx.DiGraph)
        for e1, e2 in T.edges:
            T[e1][e2]["cost"] = self.vcosts[e1]
        return T


class RRTinformed(RRTvec):
    def __init__(self, world, n, r_goal):
        super().__init__(world, n)
        self.r_goal = r_goal

    @staticmethod
    def unitball():
        r = np.random.uniform(0.0, 1.0)
        theta = 2 * np.pi * np.random.uniform(0, 1)
        x = sqrt(r) * cos(theta)
        y = sqrt(r) * sin(theta)
        unif = np.array([x, y])
        return unif

    def sample_ellipse(self, xstart, xgoal, cmin, cbest, clamp=True):
        xcent = (xstart + xgoal) / 2
        CL = self.get_ellipse_xform(xstart, xgoal, cmin, cbest)
        xball = self.unitball()
        x, y = tuple(np.dot(CL, xball) + xcent)

        if clamp:
            # clamp to finite world
            x = int(max(0, min(self.world.shape[0] - 1, x)))
            y = int(max(0, min(self.world.shape[1] - 1, y)))
        return np.array((x, y))

    def rotation_to_world_frame(self, xstart, xgoal):
        """calculate the rotation matrix from the world-frame to the frame given
        by the hyperellipsoid with focal points at xf1=xstart and xf2=xgoal. a unit
        ball multiplied by this matrix will produce an oriented ellipsoid with those
        focal points."""
        a1 = np.atleast_2d((xgoal - xstart) / norm(xgoal, xstart))
        M = np.outer(a1, np.atleast_2d([1, 0]))
        U, _, V = svd(M)
        return U @ np.diag([det(U), det(V)]) @ V.T

    def get_ellipse_xform(self, xstart, xgoal, cmin, cbest):
        """transform vector in unit plane to ellipse plane"""
        # rotation matrix
        C = self.rotation_to_world_frame(xstart, xgoal)

        assert cbest > cmin
        r1 = sqrt(cbest * cbest - cmin * cmin) / 2
        r2 = cbest / 2

        L = np.diag([r2, r1])
        # dot rotation with scale to get final matrix
        return np.dot(C, L)

    @staticmethod
    def rad2deg(a):
        return a * 180 / np.pi

    def get_ellipse_mpl_patch(self, xstart, xgoal, cmin, cbest, color=None):
        xcent = (xgoal + xstart) / 2

        # get the rotation
        CL = self.get_ellipse_xform(xstart, xgoal, cmin, cbest)

        # apply the rotation to find major axis, minor axis, angle
        a = np.dot(CL, np.array([1, 0]))
        b = np.dot(CL, np.array([0, 1]))
        majax = 2 * np.linalg.norm(a)
        minax = 2 * np.linalg.norm(b)
        ang = self.rad2deg(np.arctan2((a)[1], (a)[0]))

        if color == None:
            color = "m"

        return Ellipse(xcent, majax, minax, ang, fill=None, ec=color)

    def least_cost(self, vsoln):
        if len(vsoln) == 1:
            # just return the single element
            return vsoln[0], self.vcosts[vsoln[0]][0]
        else:
            # find the min
            csorted = np.argmin(self.vcosts[vsoln])
            return vsoln[csorted], self.vcosts[vsoln[csorted]][0]

    def make(self, xstart, xgoal, r_rewire):
        xstart = self.clamp_arr(xstart)
        xgoal = self.clamp_arr(xgoal)
        cmin = norm(xstart, xgoal)
        vsoln = list()
        # store xstart into points
        self.points[0, :] = xstart
        # store costs
        self.vcosts[0] = 0.0
        i = 1
        if self.pbar:
            pbar = tqdm(total=self.n, desc="tree")
        while i < self.n:
            if len(vsoln) == 0:
                xnew = sample_all_free(self.free)
            else:
                vbest, cbest = self.least_cost(vsoln)
                cbest += norm(self.points[vbest], xgoal)
                xnew = self.sample_ellipse(xstart, xgoal, cmin, cbest)

                ########### plot the ellipse
                # ax = plt.gca()
                # # get color
                # ellcolor_val = float(i) / self.n
                # color = cm.get_cmap("Greys")(ellcolor_val)
                # ax.add_patch(
                #     self.get_ellipse_mpl_patch(xstart, xgoal, cmin, cbest, color=color)
                # )

            vbest = nearest(self.points, xnew)[0]
            xnearest = self.points[vbest]
            nocoll = collisionfree(self.world, xnearest, xnew)
            if nocoll[0]:
                vnew = i
                vnear = self.near(xnew, r_rewire)
                vnear = vnear.reshape(vnear.shape[0])
                cbest = self.calc_cost(vbest, xnew)

                for vn in vnear:
                    xn = self.points[vn]
                    cn = self.calc_cost(vn, xnew)
                    if cn < cbest:
                        if collisionfree(self.world, xn, xnew)[0]:
                            cbest = cn
                            vbest = vn

                # store new point
                self.points[vnew] = xnew
                self.vcosts[vnew] = cbest
                # store new edge
                self.edges[vbest, vnew] = 1

                # tree rewire
                for vn in vnear:
                    xn = self.points[vn]
                    cn = self.vcosts[vn]
                    possible_cost = self.calc_cost(vn, xnew)
                    if possible_cost < cn:
                        if collisionfree(self.world, xn, xnew)[0]:
                            parent = self.get_parent(vn)
                            if parent is not None:
                                self.edges[parent, vn] = 0
                                self.edges[vnew, vn] = 1
                                self.vcosts[vn] = possible_cost

                if norm(xnew, xgoal) < self.r_goal:
                    vsoln.append(vnew)

                if self.pbar:
                    pbar.update(1)
                i += 1

        T = nx.convert_matrix.from_numpy_array(self.edges, create_using=nx.DiGraph)
        for e1, e2 in T.edges:
            T[e1][e2]["cost"] = self.vcosts[e1]

        if self.pbar:
            pbar.update(1)
            pbar.close()

        return T
