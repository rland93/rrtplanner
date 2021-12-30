import numpy as np
import numba as nb
from math import sqrt
from tqdm import tqdm
import networkx as nx
from typing import Tuple
from matplotlib import cm
from matplotlib.patches import Ellipse

############# RRT BASE CLASS ##################################################


class RRT(object):
    """base class containing common RRT methods"""

    def __init__(self, world: np.ndarray, n: int, every=10, pbar=True):
        # whether to display a progress bar
        self.pbar = pbar
        # every n tries, attempt to go to goal
        self.every = every
        # array containing vertex points
        self.n = n
        # world free space
        self.free = np.argwhere(world == 0)
        self.world = world
        # set of sampled points
        self.sampled = set()

    @staticmethod
    def near(points: np.ndarray, x: np.ndarray) -> np.ndarray:
        # vector from x to all points
        p2x = points - x
        # norm of that vector
        dist = np.linalg.norm(p2x, axis=1)
        # sorted norms
        sorted = np.argsort(dist)
        return sorted

    @staticmethod
    def within(points: np.ndarray, x: np.ndarray, r: float) -> np.ndarray:
        # vector from x to all points
        p2x = points - x
        # dot dist with self to get r2
        d2 = p2x[:, 0] * p2x[:, 0] + p2x[:, 1] * p2x[:, 1]
        # get indices of points within r2
        near_idx = np.argwhere(d2 < r * r)
        return np.atleast_1d(np.squeeze(near_idx))

    @staticmethod
    @nb.njit(fastmath=True)
    def r2norm(x):
        return sqrt(x[0] * x[0] + x[1] * x[1])

    @staticmethod
    @nb.njit()
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
        while True:
            if world[x0, y0] == 1:
                return False
            elif x0 == x1 and y0 == y1:
                return True
            else:
                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    x0 += sx
                if e2 <= dx:
                    err += dx
                    y0 += sy

    def sample_all_free(self):
        """sample uniformly from free space in the world."""
        return self.free[np.random.choice(self.free.shape[0])]

    def make(self, xstart: np.ndarray, xgoal: np.ndarray):
        raise NotImplementedError

    def set_world(self, world: np.ndarray):
        """update world with new world"""
        self.world = world
        self.free = np.argwhere(self.world == 0)

    def set_n(self, n: int):
        """update n"""
        self.n = n

    def set_every(self, every: int):
        """update every"""
        self.every = every


############# RRT STANDARD ####################################################


class RRTStandard(RRT):
    def __init__(
        self,
        world: np.ndarray,
        n: int,
        every=100,
        pbar=True,
    ):
        super().__init__(world, n, every=every, pbar=pbar)

    def cost(
        self, vcosts: np.ndarray, points: np.ndarray, v: int, x: np.ndarray
    ) -> float:
        return vcosts[v] + self.r2norm(points[v] - x)

    def make(self, xstart: np.ndarray, xgoal: np.ndarray) -> Tuple[nx.DiGraph, int]:
        points = np.full((self.n, 2), dtype=int, fill_value=1e4)
        vcosts = np.full((self.n,), fill_value=np.inf)
        edges, parents = {}, {}
        points[0] = xstart
        vcosts[0] = 0
        parents[0] = None
        i, j = 0, 1
        reached_goal = False
        if self.pbar:
            pbar = tqdm(total=self.n)

        while i < self.n:
            if self.pbar:
                pbar.update(1)
            xnew = self.sample_all_free()
            vnearest = self.near(points, xnew)[0]
            xnearest = points[vnearest]
            nocoll = self.collisionfree(self.world, xnearest, xnew)
            if nocoll and tuple(xnew) not in self.sampled:
                self.sampled.add(tuple(xnew))
                # check least cost path to xnew
                vbest = vnearest
                # store new point
                vnew = j
                points[vnew] = xnew
                vcosts[vnew] = self.cost(vcosts, points, vbest, xnew)
                # store new edge
                edges[vnew] = vbest
                parents[vbest] = vnew
                j += 1
            i += 1

        # throw away empties
        points = points[:j]
        vcosts = vcosts[:j]
        # complete, now try to find least cost to goal
        dists = np.linalg.norm(points - xgoal, axis=1)
        # add dist to each point's cost
        for i in np.argsort(vcosts + dists):
            if self.collisionfree(self.world, points[i], xgoal):
                vgoal = points.shape[0]
                reached_goal = True
                edges[vgoal] = i
                parents[i] = vgoal
                break

        # get nearest if not at goal
        if not reached_goal:
            vgoal = self.near(points, xgoal)[0]

        # build graph
        T = nx.DiGraph()
        T.add_node(vgoal, pt=xgoal)
        for i, p in enumerate(points):
            T.add_node(i, pt=p)
            T.nodes[i]["pt"]

        for e2, e1 in edges.items():
            if e1 is not None:
                if e2 == points.shape[0]:
                    p1, p2 = points[e1], xgoal
                else:
                    p1, p2 = points[e1], points[e2]
                dist = self.r2norm(p2 - p1)
                T.add_edge(e1, e2, dist=dist)
        return T, vgoal


########### RRT STAR ##########################################################


class RRTStar(RRT):
    def __init__(
        self, world: np.ndarray, n: int, r_rewire: float, every=100, pbar=True
    ):
        super().__init__(world, n, every=every, pbar=pbar)
        self.r_rewire = r_rewire

    def cost(
        self,
        vcosts: np.ndarray,
        points: np.ndarray,
        v: int,
        x: np.ndarray,
    ) -> float:
        return vcosts[v] + self.r2norm(points[v] - x)

    def make(self, xstart: np.ndarray, xgoal: np.ndarray):
        points = np.full((self.n, 2), dtype=int, fill_value=1e4)
        vcosts = np.full((self.n,), fill_value=np.inf)
        edges, parents = {}, {}
        points[0] = xstart
        vcosts[0] = 0
        parents[0] = None
        i, j = 0, 1
        reached_goal = False
        if self.pbar:
            pbar = tqdm(total=self.n)

        while i < self.n:
            if self.pbar:
                pbar.update(1)

            xnew = self.sample_all_free()
            vnearest = self.near(points, xnew)[0]
            xnearest = points[vnearest]

            nocoll = self.collisionfree(self.world, xnearest, xnew)
            if nocoll and tuple(xnew) not in self.sampled:
                self.sampled.add(tuple(xnew))

                # check least cost path to xnew
                vbest = vnearest
                cbest = self.cost(vcosts, points, vbest, xnew)
                vnear = self.within(points, xnew, self.r_rewire)

                for vn in vnear:
                    xn = points[vn]
                    cn = self.cost(vcosts, points, vn, xnew)
                    if cn < cbest:
                        if self.collisionfree(self.world, xn, xnew):
                            vbest = vn
                            cbest = cn

                # store new point
                vnew = j
                points[vnew] = xnew
                vcosts[vnew] = cbest
                # store new edge
                edges[vnew] = vbest
                parents[vbest] = vnew

                # tree rewire
                for vn in vnear:
                    xn = points[vn]
                    cn = vcosts[vn]
                    cmaybe = self.cost(vcosts, points, vn, xnew)
                    if cmaybe < cn:
                        if self.collisionfree(self.world, xn, xnew):
                            parent = parents[vn]
                            if parent is not None:
                                edges[parent] = None
                                edges[vnew] = vn
                                parents[vn] = vnew
                                vcosts[vn] = cmaybe
                j += 1
            i += 1

        # throw away empties
        points = points[:j]
        vcosts = vcosts[:j]
        # complete, now try to find least cost to goal
        dists = np.linalg.norm(points - xgoal, axis=1)
        # add dist to each point's cost
        for i in np.argsort(vcosts + dists):
            if self.collisionfree(self.world, points[i], xgoal):
                vgoal = points.shape[0]
                reached_goal = True
                edges[vgoal] = i
                parents[i] = vgoal
                break

        # get nearest if not at goal
        if not reached_goal:
            vgoal = self.near(points, xgoal)[0]

        # build graph
        T = nx.DiGraph()
        T.add_node(vgoal, pt=xgoal)
        for i, p in enumerate(points):
            T.add_node(i, pt=p)
            T.nodes[i]["pt"]

        for e2, e1 in edges.items():
            if e1 is not None:
                if e2 == points.shape[0]:
                    p1, p2 = points[e1], xgoal
                else:
                    p1, p2 = points[e1], points[e2]
                dist = self.r2norm(p2 - p1)
                T.add_edge(e1, e2, dist=dist)
        return T, vgoal


########### RRT STAR INFORMED #################################################


class RRTStarInformed(RRT):
    def __init__(self, world, n, r_rewire, r_goal, every=100, pbar=True):
        super().__init__(world, n, every=every, pbar=pbar)
        self.r_rewire = r_rewire
        self.r_goal = r_goal
        # store the ellipses for plotting later
        self.ellipses = {}

    @staticmethod
    def unitball():
        """draw a point from a uniform distribution bounded by the ball:
        U(x1, x2) ~ 1 > (x1)^2 + (x2)^2"""
        r = np.random.uniform(0, 1)
        theta = 2 * np.pi * np.random.uniform(0, 1)
        x = np.sqrt(r) * np.cos(theta)
        y = np.sqrt(r) * np.sin(theta)
        unif = np.array([x, y])
        return unif

    def sample_ellipse(self, xstart, xgoal, c, clamp=True):
        xcent = (xstart + xgoal) / 2
        CL = self.get_ellipse_xform(xstart, xgoal, c)
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
        a1 = np.atleast_2d((xgoal - xstart) / np.linalg.norm(xgoal - xstart))
        M = np.outer(a1, np.atleast_2d([1, 0]))
        U, _, V = np.linalg.svd(M)
        return U @ np.diag([np.linalg.det(U), np.linalg.det(V)]) @ V.T

    def get_ellipse_xform(self, xstart, xgoal, cmax):
        """transform vector in unit plane to ellipse plane"""
        # rotation matrix
        C = self.rotation_to_world_frame(xstart, xgoal)
        # scale by major axis r1, minor axis r2.
        r1 = cmax / 2
        d2 = np.dot((xstart - xgoal).T, (xstart - xgoal))
        r2 = np.sqrt(abs(cmax * cmax - d2)) / 2
        L = np.diag([r1, r2])
        # dot rotation with scale to get final matrix
        return np.dot(C, L)

    @staticmethod
    def least_cost(vcosts, vsoln):
        if len(vsoln) == 1:
            return vsoln[0], vcosts[vsoln[0]]
        else:
            idx_cmin = np.argmin(vcosts[vsoln])
            return vsoln[idx_cmin], vcosts[vsoln[idx_cmin]]

    @staticmethod
    def rad2deg(a):
        return a * 180 / np.pi

    def get_ellipse_for_plt(self, xstart, xgoal, cmax):
        xcent = (xgoal + xstart) / 2
        # get the rotation
        CL = self.get_ellipse_xform(xstart, xgoal, cmax)
        # apply the rotation to find major axis, minor axis, angle
        a = np.dot(CL, np.array([1, 0]))
        b = np.dot(CL, np.array([0, 1]))
        majax = 2 * np.linalg.norm(a)
        minax = 2 * np.linalg.norm(b)
        ang = self.rad2deg(np.arctan2((a)[1], (a)[0]))
        return xcent, majax, minax, ang

    def cost(self, vcosts, points, v, x):
        return vcosts[v] + self.r2norm(points[v] - x)

    def make(self, xstart: np.ndarray, xgoal: np.ndarray):

        vsoln = []
        cmin = self.r2norm(xgoal - xstart)

        points = np.full((self.n, 2), dtype=int, fill_value=1e4)
        vcosts = np.full((self.n,), fill_value=np.inf)
        edges, parents = {}, {}
        points[0] = xstart
        vcosts[0] = 0
        parents[0] = None
        i, j = 0, 1
        reached_goal = False
        if self.pbar:
            pbar = tqdm(total=self.n)

        while i < self.n:
            if self.pbar:
                pbar.update(1)

            # either sample from free or sample from ellipse depending on vsoln
            if len(vsoln) == 0:
                xnew = self.sample_all_free()
            else:
                vbest, cbest = self.least_cost(vcosts, list(vsoln))
                cbest += self.r2norm(xgoal - points[vbest])
                xnew = self.sample_ellipse(xstart, xgoal, cbest)
                self.ellipses[j] = self.get_ellipse_for_plt(xstart, xgoal, cbest)

            vnearest = self.near(points, xnew)[0]
            xnearest = points[vnearest]

            nocoll = self.collisionfree(self.world, xnearest, xnew)
            if nocoll and tuple(xnew) not in self.sampled:
                self.sampled.add(tuple(xnew))

                # check least cost path to xnew
                vbest = vnearest
                cbest = self.cost(vcosts, points, vbest, xnew)
                vnear = self.within(points, xnew, self.r_rewire)

                for vn in vnear:
                    xn = points[vn]
                    cn = self.cost(vcosts, points, vn, xnew)
                    if cn < cbest:
                        if self.collisionfree(self.world, xn, xnew):
                            vbest = vn
                            cbest = cn

                # store new point
                vnew = j
                points[vnew] = xnew
                vcosts[vnew] = cbest
                # store new edge
                edges[vnew] = vbest
                parents[vbest] = vnew

                # tree rewire
                for vn in vnear:
                    xn = points[vn]
                    cn = vcosts[vn]
                    cmaybe = self.cost(vcosts, points, vn, xnew)
                    if cmaybe < cn:
                        if self.collisionfree(self.world, xn, xnew):
                            parent = parents[vn]
                            if parent is not None:
                                edges[parent] = None
                                edges[vnew] = vn
                                parents[vn] = vnew
                                vcosts[vn] = cmaybe
                # check goal
                if self.r2norm(xnew - xgoal) < self.r_goal:
                    vsoln.append(vnew)

                j += 1
            i += 1

        # throw away empties
        points = points[:j]
        vcosts = vcosts[:j]
        # complete, now try to find least cost to goal
        dists = np.linalg.norm(points - xgoal, axis=1)
        # add dist to each point's cost
        for i in np.argsort(vcosts + dists):
            if self.collisionfree(self.world, points[i], xgoal):
                vgoal = points.shape[0]
                reached_goal = True
                edges[vgoal] = i
                parents[i] = vgoal
                break

        # get nearest if not at goal
        if not reached_goal:
            vgoal = self.near(points, xgoal)[0]

        # build graph
        T = nx.DiGraph()
        T.add_node(vgoal, pt=xgoal)
        for i, p in enumerate(points):
            T.add_node(i, pt=p)
            T.nodes[i]["pt"]

        for e2, e1 in edges.items():
            if e1 is not None:
                if e2 == points.shape[0]:
                    p1, p2 = points[e1], xgoal
                else:
                    p1, p2 = points[e1], points[e2]
                dist = self.r2norm(p2 - p1)
                T.add_edge(e1, e2, dist=dist)
        return T, vgoal


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from world_gen import make_world, get_rand_start_end

    w, h = 512, 512
    world = make_world((w, h), (4, 4))
    xstart, xgoal = get_rand_start_end(world)

    fig, ax = plt.subplots()
    ax.imshow(~world.T, origin="lower", cmap="gray")
    rrts = RRTStarInformed(world, 2000, 256, 15, every=25, pbar=True)
    T, gv = rrts.make(xstart, xgoal)

    lc = []
    for (e1, e2) in T.edges:
        p1, p2 = T.nodes[e1]["pt"], T.nodes[e2]["pt"]
        lc.append([p1, p2])

    ax.add_collection(LineCollection(lc, colors="b"))
    plt.show()
