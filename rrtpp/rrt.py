from scipy.spatial.distance import euclidean
import numpy as np
import uuid
import networkx as nx
from collections import deque
from typing import Union, overload
from scipy.linalg import svd, det, norm


def dotself(u):
    return np.dot(u, u)


def make_pos(T: nx.DiGraph) -> dict:
    """keys are nodes, values are positions"""
    pos = {}
    for n in T.nodes:
        pos[n] = T.nodes[n]["point"]
    return pos


def get_rand_start_end(world, bias=True):
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
    def __init__(self, start, goal, k):
        self.start = start
        self.goal = goal
        self.k = k

    def nearest_nodes(self, T: nx.DiGraph, x: np.array) -> list:
        """Get nearest nodes to point x

        Parameters
        ----------
        T : nx.DiGraph
            existing graph
        x : np.array
            shape (2,) existing point

        Returns
        -------
        list
            list of nodes
        """

        def distance(u: uuid.UUID):
            x1 = T.nodes[u]["point"]
            x2 = x
            return dotself(x1 - x2)

        return sorted([n for n in T.nodes], key=distance)

    def get_end_node(
        self, T: nx.DiGraph, world: np.ndarray, end_point: np.ndarray
    ) -> uuid.uuid4:
        """make or get the end node. if the end point is covered by obstacles, get the nearest node
        to it in the tree.

        Parameters
        ----------
        T : nx.DiGraph
            The world graph
        world : np.ndarray
            MxN world
        end_point : np.ndarray
            shape (2,) end point

        Returns
        -------
        uuid.uuid4
            id of the node
        """
        v_nearest = self.nearest_nodes(T, end_point)[0]
        if not self.collision(T, world, v_nearest, end_point):
            end_node = uuid.uuid4()
            T.add_node(end_node, point=end_point, active=True)
            T.add_edge(
                v_nearest,
                end_node,
                dist=self.dist_nodes(T, end_node, v_nearest),
                active=True,
            )
        else:
            end_node = v_nearest
        return end_node

    def dist_nodes(self, T: nx.DiGraph, u: uuid.uuid4, v: uuid.uuid4) -> float:
        """get euclidean distance between two nodes

        Parameters
        ----------
        T : nx.DiGraph
            world graph
        u : uuid.uuid4
            id of node 1
        v : uuid.uuid4
            id of node 2

        Returns
        -------
        float
            euclidean distance
        """
        p1 = T.nodes[u]["point"]
        p2 = T.nodes[v]["point"]
        return euclidean(p1, p2)

    def collision(
        self,
        T: nx.DiGraph,
        world,
        v1: Union[uuid.uuid4, np.array],
        v2: Union[uuid.uuid4, np.array],
    ) -> bool:
        """Calculate a collision on T.

        Parameters
        ----------
        T : nx.DiGraph
            world graph
        world : np.array
            MxN array representing the world
        v1 : Union[uuid.uuid4, np.array]
            node id or point of vertex 1
        v2 : Union[uuid.uuid4, np.array]
            node id or point of vertex 2

        Returns
        -------
        bool
            True if there is a collision

        Raises
        ------
        TypeError
            if v1 is not node id or point
        TypeError
            if v2 is not node id or point
        """
        if type(v1) == np.ndarray:
            x0 = v1[0]
            y0 = v1[1]
        elif type(v1) == uuid.UUID:
            x0 = T.nodes[v1]["point"][0]
            y0 = T.nodes[v1]["point"][1]
        else:
            raise TypeError("v1")
        if type(v2) == np.ndarray:
            x1 = v2[0]
            y1 = v2[1]
        elif type(v2) == uuid.UUID:
            x1 = T.nodes[v2]["point"][0]
            y1 = T.nodes[v2]["point"][1]
        else:
            raise TypeError("v2")

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
            if x0 == x1 and y0 == y1:
                return False
            # compare
            if world[x0, y0] == 1:
                return True
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def near(self, T: nx.DiGraph, p: np.array, r: float):
        """get nearby nodes of point p

        Parameters
        ----------
        T : nx.DiGraph
            world graph
        p : np.array
            point to check
        r : float
            within radius

        Returns
        -------
        list of nodes
            list of nearby nodes
        """
        r2 = r * r
        within = []
        for n in T.nodes:
            if dotself(T.nodes[n]["point"] - p) < r2:
                within.append(n)
        return within

    def set_start(self, start):
        self.start = start

    def set_goal(self, goal):
        self.goal = goal

    def path(self, T: nx.DiGraph, start: uuid.uuid4, end: uuid.uuid4) -> list:
        """get path from one node to another, in the form of a list of nodes comprising the path

        Parameters
        ----------
        T : nx.DiGraph
            world graph
        start : uuid.uuid4
            start node
        end : uuid.uuid4
            end node

        Returns
        -------
        list
            list of nodes making up the path
        """
        node_path = nx.shortest_path(T, source=start, target=end, weight="dist")
        point_path = np.array([T.nodes[n]["point"] for n in node_path])
        return point_path

    def random_sample(self, world, free, sample=None):
        return free[np.random.choice(free.shape[0])]


class RRTstandard(RRT):
    def __init__(self, start, goal, k):
        super().__init__(start, goal, k)

    def make(self, world: np.array):
        free = np.argwhere(world == 0)
        T = nx.DiGraph()
        start_node = uuid.uuid4()
        T.add_node(start_node, point=self.start, active=True)
        k = 0
        tries = 0
        while k < self.k:
            if tries > self.k * 3:
                break
            rand_idx = self.random_sample(world, free)
            x_rand = free[rand_idx]
            x_new = x_rand
            v_nearest = self.nearest_nodes(T, x_new)[0]
            if not self.collision(T, world, v_nearest, x_new):
                new_node = uuid.uuid4()
                T.add_node(new_node, point=x_new, active=True)
                T.add_edge(
                    v_nearest,
                    new_node,
                    dist=self.dist_nodes(T, v_nearest, new_node),
                    active=True,
                )
                k += 1
            tries += 1
        end_node = self.get_end_node(T, world, self.goal)
        return T, start_node, end_node


class RRTstar(RRT):
    """RRTstar is a variant of RRT that rewires the tree according to a cost heuristic
    every time a new node is added. tree rewirings improve the relationship that new
    nodes have to the rest of the tree, producing more optimal paths at the expense
    of increased computational cost

    Parameters
    ----------
    start : np.array
        start point
    goal : np.array
        goal point
    k : int
        no. of sample points
    dx : float
        local rewire radius
    """

    def __init__(self, start: np.array, goal: np.array, k: int, dx: float):
        super().__init__(start, goal, k)
        self.dx = dx

    def get_cost(self, T: nx.DiGraph, v: uuid.uuid4, x: np.array = None):
        """get cost of a node `v`. If `x` is passed, get the cost to that point

        Parameters
        ----------
        T : nx.DiGraph
            world graph
        v : uuid.uuid4
            node to get cost of
        x : np.array, optional
            if passed, get cost to this point, by default None

        Returns
        -------
        float
            cost
        """
        if x is None:
            return T.nodes[v]["cost"]
        else:
            dist = euclidean(T.nodes[v]["point"], x)
            return T.nodes[v]["cost"] + dist

    def make(self, world: np.array):
        """Make an RRTstar graph

        Parameters
        ----------
        world : np.array
            MxN world

        Returns
        -------
        nx.DiGraph, uuid.uuid4, uuid.uuid4
            The finished graph, the start node in the graph, the end node in the graph.
        """
        free = np.argwhere(world == 0)
        T = nx.DiGraph()
        startn = uuid.uuid4()
        T.add_node(startn, point=self.start, active=True, cost=0.0)
        k = 0
        tries = 0
        while k < self.k:
            k += 1
            tries += 1
            if tries > self.k * 2:
                break
            x = self.random_sample(world, free)
            v_nearest = self.nearest_nodes(T, x)[0]
            if not self.collision(T, world, v_nearest, x):
                newn = uuid.uuid4()
                # get minimum cost paths within radius
                nodes_near = self.near(T, x, self.dx)
                v_min = v_nearest
                c_min = self.get_cost(T, v_nearest, x)
                for nearby in nodes_near:
                    coll = self.collision(T, world, nearby, x)
                    cost = self.get_cost(T, nearby, x)
                    if (not coll) and (cost < c_min):
                        v_min = nearby
                        c_min = cost
                # rewire the tree
                dist = euclidean(T.nodes[v_min]["point"], x)
                # add new node
                T.add_node(
                    newn,
                    point=x,
                    cost=self.get_cost(T, v_min, x=x),
                    active=True,
                )
                # add new edge
                T.add_edge(v_min, newn, dist=dist, active=True)
                for nearby in nodes_near:
                    coll = not self.collision(T, world, nearby, x)
                    nearby_pt = T.nodes[nearby]["point"]
                    cost = self.get_cost(T, newn) + self.get_cost(T, newn, nearby_pt)
                    if coll and cost < self.get_cost(T, nearby):
                        v_parent = next(T.predecessors(nearby), None)
                        if v_parent is not None:
                            dist = self.dist_nodes(T, nearby, newn)
                            T.add_edge(newn, nearby, dist=dist, active=True)
                            T.remove_edge(v_parent, nearby)
        endn = self.get_end_node(T, world, self.goal)
        return T, startn, endn


class RRTstarInformed(RRT):
    def __init__(self):
        pass

    def get_cbest(self, T, Vsoln):
        if len(Vsoln) == 0:
            return None
        else:
            return min([v for v in Vsoln], key=lambda u: T.nodes[u]["cost"])

    @staticmethod
    def rotation_to_world_frame(xstart, xgoal):
        a1 = np.atleast_2d((xgoal - xstart) / euclidean(xgoal, xstart))
        B = np.dot(a1.T, np.atleast_2d([0, 1]))
        U, _, V = svd(B)
        d = np.diag([det(U), det(V)])
        rotmat = U @ d @ V.T
        return rotmat

    @staticmethod
    def unitball():
        r = np.random.uniform(0, 1)
        theta = np.pi * np.random.uniform(0, 2)
        x = np.sqrt(r) * np.cos(theta)
        y = np.sqrt(r) * np.sin(theta)
        return np.array([x, y])

    def sample(self, world, xstart, xgoal, cmax=None):
        free = np.argwhere(world == 0)
        if cmax is not None:
            cmin = dotself(xstart - xgoal) / norm(xstart - xgoal)
            xcent = (xstart + xgoal) / 2
            C = self.rotation_to_world_frame(xstart, xgoal)
            r1 = cmax / 2
            r2 = np.sqrt(np.abs(cmax * cmax - cmin * cmin)) / 2.0
            L = np.diag([r1, r2])
            # xball = self.unitball()
            xball = np.random.normal(size=(2,))
            CL = np.dot(C, L)
            x, y = tuple(np.dot(CL, xball) + xcent)
            # clamp
            x = int(max(0, min(world.shape[0] - 1, x)))
            y = int(max(0, min(world.shape[1] - 1, y)))
            point = np.array((x, y))
            return point
        else:
            return free[np.random.choice(free.shape[0])]

    def steer(self, xnearest, xrand):
        pass

    def get_cost(self, T: nx.DiGraph, v: uuid.uuid4, x: np.array = None):
        """get cost of a node `v`. If `x` is passed, get the cost to that point"""
        if x is None:
            return T.nodes[v]["cost"]
        else:
            dist = euclidean(T.nodes[v]["point"], x)
            return T.nodes[v]["cost"] + dist

    def get_parent(self, T, v):
        return next(T.predecessors(v), None)

    def collision(
        self,
        T: nx.DiGraph,
        world,
        v1: np.array,
        v2: np.array,
    ) -> bool:
        x0 = v1[0]
        y0 = v1[1]
        x1 = v2[0]
        y1 = v2[1]

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
            if x0 == x1 and y0 == y1:
                return False
            # compare
            if world[x0, y0] == 1:
                return True
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def make(self, world, xstart, xgoal, N, r, rgoal):
        i = 1
        Vsoln = set()
        T = nx.DiGraph()
        T.add_node(i, point=xstart, cost=0)

        j = 0
        while i < N:
            cbest = self.get_cbest(T, Vsoln)
            xrand = self.sample(world, xstart, xgoal, cmax=cbest)
            vnearest = self.nearest_nodes(T, xrand)[0]
            xnew = xrand
            if not self.collision(T, world, T.nodes[vnearest]["point"], xnew):
                i += 1
                vnear = self.near(T, xnew, r)
                cmin = self.get_cost(T, vnearest, xnew)
                for vn in vnear:
                    cnew = self.get_cost(T, vn, xnew)
                    if cnew < cmin:
                        if not self.collision(T, world, T.nodes[vn]["point"], xnew):
                            vnearest = vn
                            cmin = cnew

                T.add_node(i, point=xnew, cost=self.get_cost(T, vnearest, xnew))
                T.add_edge(vnearest, i)

                for vn in vnear:
                    cnear = self.get_cost(T, vn)
                    cnew = self.get_cost(T, vn, xnew)
                    if cnew < cnear:
                        if not self.collision(T, world, T.nodes[vn]["point"], xnew):
                            vparent = self.get_parent(T, vn)
                            if vparent is not None:
                                T.remove_edge(vparent, vn)
                                T.add_edge(vn, i)

                if dotself(xnew - xgoal) < dotself(rgoal):
                    Vsoln.add(i)
        return T
