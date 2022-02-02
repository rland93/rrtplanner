import numpy as np
import numba as nb
from math import sqrt
from tqdm import tqdm
import networkx as nx
from typing import Tuple, List
from collections import defaultdict


@nb.njit(fastmath=True)
def r2norm(x):
    return sqrt(x[0] * x[0] + x[1] * x[1])


############# RRT BASE CLASS ##################################################


class RRT(object):
    """base class containing common RRT methods"""

    def __init__(
        self,
        world: np.ndarray,
        n: int,
        costfn: callable = None,
        every: int = 10,
        pbar: bool = True,
    ):
        # whether to display a progress bar
        self.pbar = pbar
        # every n tries, attempt to go to goal
        self.every = every
        # array containing vertex points
        self.n = n
        # world free space
        self.free = np.argwhere(world == 0)
        self.world = world

        # define simple r2norm cost function if
        # no cost function is passed in

        if costfn is None:

            def costfn(
                vcosts: np.ndarray,
                points: np.ndarray,
                v: int,
                x: np.ndarray,
            ) -> float:
                return vcosts[v] + r2norm(points[v] - x)

        self.cost = costfn
        self.not_a_point = [np.inf, np.inf]
        self.not_a_dist = np.inf

    def route2gv(self, T: nx.DiGraph, gv) -> List[int]:
        return nx.shortest_path(T, source=0, target=gv, weight="dist")

    def path_points(self, T: nx.DiGraph, path: list) -> np.ndarray:
        lines = []
        for i in range(len(path) - 1):
            lines.append([T.nodes[path[i]]["pt"], T.nodes[path[i + 1]]["pt"]])
        return np.array(lines)

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
            if world[x0, y0] != 0:
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

    def go2goal(self, vcosts, points, xgoal, j, children, parents):
        # cost for all existing points
        costs = np.empty(vcosts.shape)
        for i in range(points.shape[0]):
            costs[i] = self.cost(vcosts, points, i, xgoal)

        found_goal = False
        for idx in np.argsort(costs):
            if self.collisionfree(self.world, points[idx], xgoal):
                vgoal = j
                points = np.concatenate((points, xgoal[np.newaxis, :]), axis=0)
                vcosts = np.concatenate((vcosts, [costs[idx]]), axis=0)
                points[vgoal] = xgoal
                vcosts[vgoal] = costs[idx]
                children[idx].append(vgoal)
                parents[vgoal] = idx
                found_goal = True
                break
        if not found_goal:
            # shrink points array
            dists = np.linalg.norm(points - xgoal)
            vgoal = np.argmin(dists)
        return vgoal, children, parents, points, vcosts

    def build_graph(self, vgoal, points, parents, vcosts):
        assert points.max() < self.not_a_point[0] - 1
        # build graph
        T = nx.DiGraph()
        T.add_node(vgoal, pt=points[vgoal])
        for i, p in enumerate(points):
            T.add_node(i, pt=p)

        for child, parent in parents.items():
            if parent is not None:
                p1, p2 = points[parent], points[child]
                dist = r2norm(p2 - p1)
                T.add_edge(parent, child, dist=dist, cost=vcosts[child])
        return T


############# RRT STANDARD ####################################################


class RRTStandard(RRT):
    def __init__(
        self,
        world: np.ndarray,
        n: int,
        costfn: callable = None,
        every=100,
        pbar=True,
    ):
        super().__init__(world, n, costfn=costfn, every=every, pbar=pbar)

    def make(self, xstart: np.ndarray, xgoal: np.ndarray) -> Tuple[nx.DiGraph, int]:
        sampled = set()
        points = np.full((self.n, 2), dtype=int, fill_value=self.not_a_point)
        vcosts = np.full((self.n,), fill_value=self.not_a_dist)
        parents, children = {}, defaultdict(list)
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
            nocoll = self.collisionfree(self.world, xnearest, xnew)
            if nocoll and tuple(xnew) not in sampled and j != self.n:
                sampled.add(tuple(xnew))
                # check least cost path to xnew
                vbest = vnearest
                # store new point
                vnew = j
                points[vnew] = xnew
                vcosts[vnew] = self.cost(vcosts, points, vbest, xnew)
                # store new edge
                children[vbest].append(vnew)
                parents[vnew] = vbest
                j += 1
            i += 1

        # go to goal if possible
        vgoal, children, parents, points, vcosts = self.go2goal(
            vcosts, points, xgoal, j, children, parents
        )

        # build graph
        T = self.build_graph(vgoal, points, parents, vcosts)

        return T, vgoal


########### RRT STAR ##########################################################


class RRTStar(RRT):
    def __init__(
        self,
        world: np.ndarray,
        n: int,
        r_rewire: float,
        costfn: callable = None,
        every=100,
        pbar=True,
    ):
        super().__init__(world, n, costfn=costfn, every=every, pbar=pbar)
        self.r_rewire = r_rewire

    def make(self, xstart: np.ndarray, xgoal: np.ndarray):
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

            nocoll = self.collisionfree(self.world, xnearest, xnew)
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
                        if self.collisionfree(self.world, xn, xnew):
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
                        if self.collisionfree(self.world, xn, xnew):
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


########### RRT STAR INFORMED #################################################


class RRTStarInformed(RRT):
    def __init__(
        self,
        world: np.ndarray,
        n: int,
        r_rewire: float,
        r_goal: float,
        costfn: callable = None,
        every: int = 100,
        pbar: bool = True,
    ):
        super().__init__(world, n, costfn=costfn, every=every, pbar=pbar)
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
        try:
            U, _, V = np.linalg.svd(M)
        except np.linalg.LinAlgError:
            # handle SVD not converging
            U, _, V = np.linalg.svd(M, full_matrices=False)
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

    def get_ellipse_for_plt(
        self, xstart, xgoal, cmax
    ) -> Tuple[np.ndarray, float, float, float]:
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

    def make(self, xstart: np.ndarray, xgoal: np.ndarray):

        vsoln = []
        sampled = set()
        points = np.full((self.n, 2), dtype=int, fill_value=self.not_a_point)
        vcosts = np.full((self.n,), fill_value=self.not_a_dist)
        parents, children = {}, defaultdict(list)
        points[0] = xstart
        vcosts[0] = 0
        parents[0] = None
        i, j = 0, 1
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
                cbest += r2norm(xgoal - points[vbest])
                xnew = self.sample_ellipse(xstart, xgoal, cbest)
                self.ellipses[j] = self.get_ellipse_for_plt(xstart, xgoal, cbest)

            vnearest = self.near(points, xnew)[0]
            xnearest = points[vnearest]

            nocoll = self.collisionfree(self.world, xnearest, xnew)
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
                        if self.collisionfree(self.world, xn, xnew):
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
                        if self.collisionfree(self.world, xn, xnew):
                            parent = parents[vn]
                            if parent is not None:
                                children[parent].remove(vn)
                                parents[vn] = vnew
                                vcosts[vn] = cmaybe
                # check goal
                if r2norm(xnew - xgoal) < self.r_goal:
                    vsoln.append(vnew)

                j += 1
            i += 1

        # go to goal if possible
        vgoal, children, parents, points, vcosts = self.go2goal(
            vcosts, points, xgoal, j, children, parents
        )

        # build graph
        T = self.build_graph(vgoal, points, parents, vcosts)

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
    rrts = RRTStarInformed(world, 400, 256, 15, every=25, pbar=True)
    T, gv = rrts.make(xstart, xgoal)

    lc = []
    for (e1, e2) in T.edges:
        p1, p2 = T.nodes[e1]["pt"], T.nodes[e2]["pt"]
        lc.append([p1, p2])

    ax.add_collection(LineCollection(lc, colors="b"))
    # plt.show()
