import numpy as np
import numba as nb
from matplotlib.collections import LineCollection
from math import sqrt
from tqdm import tqdm
import networkx as nx


def get_rrt_LC(T, c="grey"):
    lines = []
    for e1, e2 in T.edges():
        pt1, pt2 = T.nodes[e1]["pos"], T.nodes[e2]["pos"]
        lines.append((pt1, pt2))
    return LineCollection(lines, color=c)


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


@nb.njit(fastmath=True)
def sample_all_free(free):
    """sample uniformly from free space in the world."""
    return free[np.random.choice(free.shape[0])]


@nb.njit(fastmath=True)
def norm(u, v):
    d = u - v
    return sqrt(d[0] * d[0] + d[1] * d[1])


@nb.njit(fastmath=True)
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
            return False
        elif x0 == x1 and y0 == y1:
            return True
        else:
            x0_old, y0_old = x0, y0
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy


@nb.njit(fastmath=True)
def r2norm(x):
    return sqrt(x[0] * x[0] + x[1] * x[1])


def nearest(points, x):
    near = []
    for p in points:
        near.append(r2norm(p - x))
    return np.argmin(near)


def near(points, x, r):
    """find idx of points within r of x"""
    near = []
    for i, p in enumerate(points):
        if r2norm(p - x) < r:
            near.append(i)
    return near


class RRTvec(object):
    """base class containing common RRT methods"""

    def __init__(self, world, n, every=10, pbar=True):
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


class RRTstar(RRTvec):
    def __init__(self, world, n, every=100, pbar=True):
        super().__init__(world, n, every=every, pbar=pbar)

    @staticmethod
    def cost(vcosts, points, v, x):
        return vcosts[v] + r2norm(points[v] - x)

    def make(self, xstart, xgoal, r_rewire):
        points, vcosts = np.full((self.n, 2), dtype=int, fill_value=1e20), np.full(
            (self.n,), fill_value=np.inf
        )
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

            xnew = sample_all_free(self.free)
            vnearest = nearest(points, xnew)
            xnearest = points[vnearest]

            nocoll = collisionfree(self.world, xnearest, xnew)
            if nocoll and tuple(xnew) not in self.sampled:
                self.sampled.add(tuple(xnew))

                # check least cost path to xnew
                vbest = vnearest
                cbest = self.cost(vcosts, points, vbest, xnew)
                vnear = near(points, xnew, r_rewire)

                for vn in vnear:
                    xn = points[vn]
                    cn = self.cost(vcosts, points, vn, xnew)
                    if cn < cbest:
                        if collisionfree(self.world, xn, xnew):
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
                        if collisionfree(self.world, xn, xnew):
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
            if collisionfree(self.world, points[i], xgoal):
                vgoal = points.shape[0]
                reached_goal = True
                edges[vgoal] = i
                parents[i] = vgoal
                break

        # get nearest if not at goal
        if not reached_goal:
            vgoal = nearest(points, xgoal)

        # build graph
        T = nx.DiGraph()
        T.add_node(vgoal, pos=xgoal)
        for i, p in enumerate(points):
            T.add_node(i, pos=p)
            T.nodes[i]["pos"]

        for e2, e1 in edges.items():
            if e1 is not None:
                if e2 == points.shape[0]:
                    p1, p2 = points[e1], xgoal
                else:
                    p1, p2 = points[e1], points[e2]
                dist = r2norm(p2 - p1)
                T.add_edge(e1, e2, dist=dist)
        return T, vgoal


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from world_gen import make_world, get_rand_start_end

    w, h = 512, 512
    world = make_world((w, h), (4, 4))
    xstart, xgoal = get_rand_start_end(world)

    rrts = RRTstar(world, 10000, every=10000, pbar=True)
    T, gv = rrts.make(xstart, xgoal, r_rewire=64)
    fig, ax = plt.subplots()
    ax.add_collection(get_rrt_LC(T))
    ax.imshow(world.T, cmap="Greys", origin="lower")
    plt.show()
