from ctypes import ArgumentError
from scipy.spatial.distance import euclidean
import numpy as np
import networkx as nx
from scipy.linalg import svd, det, norm
from tqdm import tqdm
from matplotlib.patches import Ellipse


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


class RRT(object):
    """base class containing common RRT methods"""

    def __init__(self, world, pbar=True):
        # the world
        self.world = world
        # the tree
        self.T = nx.DiGraph()
        # whether to display a pibar
        self.pbar = pbar
        # writeable data from `make` or other methods
        self.data = {}
        # start and end nodes
        self.vstart, self.vend = None, None

    def sample_all_free(self):
        """sample uniformly from free space in the world."""
        free = np.argwhere(self.world == 0)
        return free[np.random.choice(free.shape[0])]

    @staticmethod
    def d2(u, v):
        return np.dot(u - v, u - v)

    @staticmethod
    def dotself(u):
        return np.dot(u, u)

    def get_parent(self, v):
        """get first parent of node `v`. If a parent does not exist,
        return None."""
        return next(self.T.predecessors(v), None)

    def near(self, x, r):
        """get nodes within `r` of point `x`"""
        r2 = r * r
        within = []
        for n in self.T.nodes:
            if self.d2(self.T.nodes[n]["point"], x) < r2:
                within.append(n)
        return within

    def nearest_nodes(self, x) -> list:
        """get a list of the nodes sorted by distance to point `x`, nearest first"""

        def distance(u):
            x1 = self.T.nodes[u]["point"]
            x2 = x
            d = self.d2(x1, x2)
            if d != 0:
                return d
            else:
                return np.Infinity

        return sorted([n for n in self.T.nodes], key=distance)

    def update_world(self, neww):
        self.world = neww

    def collisionfree(self, a, b) -> bool:
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
            if self.world[x0, y0] == 1:
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

    def calc_cost(self, v, x=None):
        if x is not None:
            return self.T.nodes[v]["cost"] + euclidean(self.T.nodes[v]["point"], x)
        else:
            return self.T.nodes[v]["cost"]

    def reset_T(self):
        del self.T
        self.T = nx.DiGraph()

    def get_end_node(self, xgoal):
        """get the 'end' node, given a tree and an end point. The end node is either the point itself,
        if a path to it is possible, or the closest node in the tree to the end point."""
        vnearest = self.nearest_nodes(xgoal)[0]
        xnearest = self.T.nodes[vnearest]["point"]
        if self.collisionfree(xnearest, xgoal):
            v = max(self.T.nodes) + 1
            newcost = self.calc_cost(vnearest, xgoal)
            self.T.add_node(v, point=xgoal, cost=newcost)
            self.T.add_edge(vnearest, v, dist=euclidean(xnearest, xgoal), cost=newcost)
            return v
        else:
            return vnearest

    def write_dist_edges(self):
        for e1, e2 in self.T.edges(data=False):
            p1 = self.T.nodes[e1]["point"]
            p2 = self.T.nodes[e2]["point"]
            self.T[e1][e2]["dist"] = euclidean(p1, p2)

    def path(self, startv=None, endv=None, weight_attr="dist"):
        """get shortest path from one vertex to another, in the form
        of an Mx2 list of points"""
        if startv == None:
            startv = self.vstart
        if endv == None:
            endv = self.vend

        path = nx.shortest_path(self.T, source=startv, target=endv, weight=weight_attr)
        point_path = np.atleast_2d([self.T.nodes[n]["point"] for n in path])
        return point_path


class RRTStarInformed(RRT):
    def __init__(self, world, **kwargs):
        super().__init__(world, **kwargs)

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
        a1 = np.atleast_2d((xgoal - xstart) / norm(xgoal - xstart))
        M = np.outer(a1, np.atleast_2d([1, 0]))
        U, _, V = svd(M)
        return U @ np.diag([det(U), det(V)]) @ V.T

    def get_ellipse_xform(self, xstart, xgoal, cmax):
        """transform vector in unit plane to ellipse plane"""
        # rotation matrix
        C = self.rotation_to_world_frame(xstart, xgoal)
        # scale by major axis r1, minor axis r2.
        r1 = cmax / 2
        r2 = np.sqrt(abs(self.dotself(cmax) - self.d2(xstart, xgoal))) / 2
        L = np.diag([r1, r2])
        # dot rotation with scale to get final matrix
        return np.dot(C, L)

    @staticmethod
    def rad2deg(a):
        return a * 180 / np.pi

    def get_ellipse_mpl_patch(self, xstart, xgoal, cmax):
        xcent = (xgoal + xstart) / 2

        # get the rotation
        CL = self.get_ellipse_xform(xstart, xgoal, cmax)

        # apply the rotation to find major axis, minor axis, angle
        a = np.dot(CL, np.array([1, 0]))
        b = np.dot(CL, np.array([0, 1]))
        majax = 2 * norm(a)
        minax = 2 * norm(b)
        ang = self.rad2deg(np.arctan2((a)[1], (a)[0]))

        return Ellipse(xcent, majax, minax, ang, fill=None, ec="m")

    def iter_vs_with_costs(self, vs, rel_x=None, check_collision=True):
        """generator expression for getting cost of addding a vertex
        at `rel_x` to each vertex in vs. yields (vertex, point, cost to add)
        tuple. if check_collision is true, only yield if there is no collision"""
        for v in vs:
            x = self.T.nodes[v]["point"]
            c = self.calc_cost(v, x=rel_x)
            if check_collision:
                if self.collisionfree(x, rel_x):
                    yield (v, x, c)
            else:
                yield (v, x, c)

    def least_cost(self, V):
        """get least cost vertex and its cost from a collection `V` of vertices in T"""
        return min([(v, self.T.nodes[v]["cost"]) for v in V], key=lambda t: t[1])

    def make(self, xstart, xgoal, N, r_rewire, r_goal) -> tuple:
        """make rrtstar informed"""
        Vsoln = set()
        self.reset_T()
        i = 1
        vstart = 0
        self.T.add_node(vstart, point=xstart, cost=0.0)
        if self.pbar:
            pbar = tqdm(total=N)
        while i < N:
            if len(Vsoln) == 0:
                xnew = self.sample_all_free()
            else:
                vbest, cbest = self.least_cost(Vsoln)
                cbest += euclidean(self.T.nodes[vbest]["point"], xgoal)
                xnew = self.sample_ellipse(xstart, xgoal, cbest)

            vnearest = self.nearest_nodes(xnew)[0]
            xnearest = self.T.nodes[vnearest]["point"]

            if self.collisionfree(xnearest, xnew):
                # get all near vertices in rewire range; nearest gets appended if
                # there are no vertices within range
                vnear = self.near(xnew, r_rewire)
                vnear.append(vnearest)
                # an edge from `vbest`->`i` (where `i` is the new vertex)
                # where `xbest` is the point of `vbest`, has cost `cost`
                vbest, _, cost = min(
                    ((v, x, c) for (v, x, c) in self.iter_vs_with_costs(vnear, xnew)),
                    key=lambda t: t[2],
                )
                # add node
                self.T.add_node(i, point=xnew, cost=cost)
                self.T.add_edge(vbest, i, cost=cost)
                # update progress bar when new node is added
                if self.pbar:
                    pbar.update(1)

                # now, we're going to rewire in the area around the
                # new vertex again. This time, we calculate the cost to go from
                # the new vertex to all of the other vertices. If that rewired cost,
                # from that vertex to the new vertex, drop the current and rewire.
                for (v, x, c) in self.iter_vs_with_costs(vnear, check_collision=False):
                    rewired_cost = self.calc_cost(i, x)
                    if rewired_cost < c:
                        if self.collisionfree(x, xnew):
                            vparent = self.get_parent(v)
                            if vparent is not None:
                                self.T.remove_edge(vparent, v)
                                self.T.nodes[v]["cost"] = rewired_cost
                                self.T.add_edge(i, v, cost=rewired_cost)
                # finally, check if we are within the goal radius.
                # nodes within the goal radius are used to narrow the search
                # space to only those paths which could improve the best
                # path ending within the goal radius.
                if self.d2(xnew, xgoal) < r_goal * r_goal:
                    Vsoln.add(i)
                i += 1
        if self.pbar:
            pbar.update(1)
            pbar.close()

        if len(Vsoln) > 0:
            vend, cbest = min(
                [(v, self.T.nodes[v]["cost"]) for v in Vsoln], key=lambda t: t[1]
            )
            cbest += euclidean(self.T.nodes[vend]["point"], xgoal)
            ell = self.get_ellipse_mpl_patch(xstart, xgoal, cbest)
        else:
            ell = Ellipse((0, 0), 1, 1, 0)
            vend = self.nearest_nodes(xgoal)[0]
        self.data["ellipse"] = ell

        self.vstart = vstart
        self.vend = vend

        return vstart, vend


class RRTStar(RRT):
    def __init__(self, world):
        super().__init__(world)

    def make(self, xstart, xgoal, N, r_rewire) -> tuple:
        """Make RRT star with `N` points from xstart to xgoal.
        Returns the tree, the start node, and the end node."""
        self.reset_T()
        i = 1
        vstart = 0
        self.T.add_node(vstart, point=xstart, cost=0.0)
        if self.pbar:
            pbar = tqdm(total=N, desc="tree")
        while i < N:
            xnew = self.sample_all_free()
            vnearest = self.nearest_nodes(xnew)
            xnearest = self.T.nodes[vnearest[0]]["point"]
            if self.collisionfree(xnearest, xnew):
                vnew = i
                vmin = vnearest[0]
                xmin = xnearest
                cmin = self.calc_cost(vmin, xnew)
                # vnear contains at least vmin
                vnear = self.near(xnew, r_rewire)
                vnear.append(vmin)
                # search for a lesser cost vertex in connection radius
                for vn in vnear:
                    xn = self.T.nodes[vn]["point"]
                    cost = self.calc_cost(vn, xnew)
                    if cost < cmin:
                        if self.collisionfree(xn, xnew):
                            xmin = xn
                            cmin = cost
                            vmin = vn
                # add new vertex and edge connecting min-cost vertex with new point
                self.T.add_node(vnew, point=xnew, cost=cmin)
                self.T.add_edge(vmin, vnew, cost=cmin)
                # rewire the tree
                for vn in vnear:
                    xn = self.T.nodes[vn]["point"]
                    cn = self.calc_cost(vn)
                    c = self.calc_cost(vn, xnew)
                    if c < cn:
                        if self.collisionfree(xn, xnew):
                            parent = self.get_parent(vn)
                            if parent is not None:
                                self.T.remove_edge(parent, vn)
                                self.T.add_edge(vnew, vn)
                if self.pbar:
                    pbar.update(1)
                i += 1
        if self.pbar:
            pbar.update(1)
            pbar.close()

        self.write_dist_edges()
        vend = self.get_end_node(xgoal)
        self.vstart = vstart
        self.vend = vend
        return vstart, vend


class RRTStandard(RRT):
    def __init__(self, world):
        super().__init__(world)
        self.T = nx.DiGraph()

    def make(self, xstart, xgoal, N) -> tuple:
        """Make RRT standard tree with `N` points from xstart to xgoal.
        Returns the tree, the start node, and the end node."""
        self.reset_T()
        i = 1
        vstart = 0
        self.T.add_node(vstart, point=xstart, cost=0)
        if self.pbar:
            pbar = tqdm(total=N)
        while i < N:
            # uniform sample over world
            xnew = self.sample_all_free()
            vnearest = self.nearest_nodes(xnew)[0]
            xnearest = self.T.nodes[vnearest]["point"]
            if self.collisionfree(xnearest, xnew) and all(xnearest != xnew):
                cost = self.calc_cost(vnearest, i)
                self.T.add_node(i, point=xnew, cost=cost)
                self.T.add_edge(vnearest, i, cost=cost)
                if self.pbar:
                    pbar.update(1)
                i += 1
        if self.pbar:
            pbar.update(1)
            pbar.close()
        # write dist of edge onto each edge
        self.write_dist_edges()
        vend = self.get_end_node(xgoal)
        self.vstart = vstart
        self.vend = vend
        return vstart, vend
