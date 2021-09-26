from sys import _xoptions
from networkx.generators.geometric import euclidean
from numpy.typing import _128Bit
from scipy.spatial.distance import euclidean
import numpy as np
from . import world_gen
import uuid
from tqdm import tqdm
import networkx as nx
from collections import deque
from typing import Union
import matplotlib.pyplot as plt


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


'''

def points_on_line(start, end, res):
    v = end - start
    # unit vector pointing from start to end
    pvec = v / np.linalg.norm(v)
    # total distance
    d = euclidean(start, end)
    # total # points
    n_pts = int(np.ceil(d / res))
    for k in range(n_pts):
        yield tuple(start + pvec * (k + 1) * res)




class RRT(object):
    def __init__(self, K, world):
        self.K = K
        self.world = world
        self.free = np.argwhere(world == 0)

    def get_rand_start_end(self, bias=True):
        """if bias, prefer points far away from one another"""
        if bias == True:
            start_i = int(np.random.beta(a=0.5, b=5) * self.free.shape[0])
            end_i = int(np.random.beta(a=5, b=0.5) * self.free.shape[0])
        else:
            start_i = np.random.choice(self.free.shape[0])
            end_i = np.random.choice(self.free.shape[0])
        start = self.free[start_i, :]
        end = self.free[end_i, :]
        return start, end

    def path(self, T, start, end):
        node_path = nx.shortest_path(T, source=start, target=end, weight="dist")
        point_path = np.array([T.nodes[n]["point"] for n in node_path])
        return point_path

    def obstacle_free(self, T, x_rand):
        # go through nearest vertices
        for v in self.nearest_vertices(x_rand, T):
            # check collision on each
            if self.collision_check(v, x_rand, T):
                continue
            else:
                return v
        return False

    def nearest(self, T, x_rand):
        def distance(u):
            x1 = T.nodes[u]["point"]
            x2 = x_rand
            return euclidean(x1, x2)

        return sorted([n for n in T.nodes], key=distance)

    def collision_check(self, v1, v2, T):
        if type(v1) == np.ndarray:
            x0 = v1[0]
            y0 = v1[1]
        elif type(v1) == uuid.UUID:
            x0 = T.nodes[v1]["point"][0]
            y0 = T.nodes[v1]["point"][1]
        else:
            print(type(v1))
        if type(v2) == np.ndarray:
            x1 = v2[0]
            y1 = v2[1]
        elif type(v2) == uuid.UUID:
            x1 = T.nodes[v2]["point"][0]
            y1 = T.nodes[v2]["point"][1]
        else:
            print(type(v2))

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
            if self.world[x0, y0] == 1:
                return True
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def near(self, T, v, r):
        """Get nodes within `r` of node `v`."""
        r2 = r * r
        return [
            n
            for n in T.nodes
            if dotself(T.nodes[v]["point"] - T.nodes[v]["point"]) < r2
        ]

    def dist_nodes(self, u, v, T):
        p1 = T.nodes[u]["point"]
        p2 = T.nodes[v]["point"]
        return euclidean(p1, p2)

    def get_end_node(self, end_point, T):
        v_nearest = self.nearest(T, end_point)[0]
        if not self.collision_check(v_nearest, end_point, T):
            end_node = uuid.uuid4()
            T.add_node(end_node, point=end_point, active=True)
            T.add_edge(
                v_nearest,
                end_node,
                dist=self.dist_nodes(end_node, v_nearest, T),
                active=True,
            )
        else:
            print(
                "WARNING: Could not find obstacle-free to end point! Try increasing no. of iterations."
            )
            end_node = v_nearest
        return end_node


class RRTStandard(RRT):
    def __init__(self, K, world):
        super().__init__(K, world)

    def build_rrt(self, start_point, end_point):
        T = nx.DiGraph()
        start_node = uuid.uuid4()
        T.add_node(start_node, point=start_point, active=True)
        pts_added = set()
        for _ in tqdm(range(self.K)):
            rand_idx = np.random.choice(self.free.shape[0])
            if rand_idx in pts_added:
                self.K += 1
                continue
            x_rand = self.free[rand_idx]
            x_new = x_rand
            v_nearest = self.nearest(T, x_new)[0]
            if not self.collision_check(v_nearest, x_new, T):
                new_node = uuid.uuid4()
                T.add_node(new_node, point=x_new, active=True)
                T.add_edge(
                    v_nearest,
                    new_node,
                    dist=self.dist_nodes(v_nearest, new_node, T),
                    active=True,
                )
            pts_added.add(rand_idx)

        end_node = self.get_end_node(end_point, T)
        return T, start_node, end_node


class RRTStar_Adaptive(object):
    def __init__(self, K, world, dx, start, goal):
        self.K = K
        self.world = world
        self.free = np.argwhere(world == 0)
        self.dx = dx
        self.T = nx.Graph()
        self.X_sampled = set()
        # make goal
        goaln = uuid.uuid4()
        self.T.add_node(goaln, point=goal, lmc=0.0, cost=10, active=True)
        self.goaln = goaln
        self.goalp = goal

        # make start
        startn = uuid.uuid4()
        self.T.add_node(
            startn, point=start, lmc=self.get_lmc(start), cost=0, active=True
        )
        self.startn = startn
        # vbot is at start
        self.v_bot = startn

        # closest node
        self.closest = startn

        self.resample()

    def update_bot_pos(self):
        nx.shortest_path(self.T, source=self.v_bot, target=self.closest, weight="dist")
        self.v_bot

    def get_pos(self):
        """keys are nodes, values are positions"""
        pos = {}
        for n in self.T.nodes:
            pos[n] = self.T.nodes[n]["point"]
        return pos

    def nearest_nodes(self, x_rand):
        """get nodes nearest to x_rand"""

        def d2(u):
            x1 = self.T.nodes[u]["point"]
            x2 = x_rand
            return dotself(x1 - x2)

        return sorted([n for n in self.T.nodes], key=d2)

    def get_cost(self, v, x=None):
        if x is None:
            return self.T.nodes[v]["cost"]
        else:
            dist = euclidean(self.T.nodes[v]["point"], x)
            return self.T.nodes[v]["cost"] + dist

    def near(self, x, dx):
        """get all nodes within dx of point x"""
        dx2 = dx * dx

        def withindx(v):
            return dotself(self.T.nodes[v]["point"] - x) < dx2

        return [v for v in self.T.nodes if withindx(v)]

    def ndist(self, u, v):
        """euclidean distance between nodes u, v"""
        p1 = self.T.nodes[u]["point"]
        p2 = self.T.nodes[v]["point"]
        return euclidean(p1, p2)

    def calculate_node_collisions(self):
        # go through edges and find collisions
        for e1, e2 in self.T.edges:
            if self.collision(e1, e2):
                self.T[e1][e2]["active"] = False
                self.T.nodes[e1]["active"] = False
                self.T.nodes[e2]["active"] = False
            else:
                self.T[e1][e2]["active"] = True
                self.T.nodes[e1]["active"] = True
                self.T.nodes[e2]["active"] = True

    def get_rand_pt(self):
        """sample until we get a novel point"""
        while True:
            rand_idx = np.random.choice(self.free.shape[0])
            if rand_idx in self.X_sampled:
                continue
            else:
                return self.free[rand_idx]

    def collision(self, a, b):
        if type(a) == np.ndarray:
            x0 = a[0]
            y0 = a[1]
        elif type(a) == uuid.UUID:
            x0 = self.T.nodes[a]["point"][0]
            y0 = self.T.nodes[a]["point"][1]
        else:
            raise TypeError("wrong type for a={}, type {}".format(a, type(a)))
        if type(b) == np.ndarray:
            x1 = b[0]
            y1 = b[1]
        elif type(b) == uuid.UUID:
            x1 = self.T.nodes[b]["point"][0]
            y1 = self.T.nodes[b]["point"][1]
        else:
            raise TypeError("wrong type for b={}, type {}".format(b, type(b)))

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
            if self.world[x0, y0] == 1:
                return True
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def get_vbot_tree(self):
        for ccs in nx.connected_components(self.T):
            if self.v_bot in ccs:
                ccs.add(self.goaln)
                # add back goal node
                self.closest = min(
                    [
                        n
                        for n in ccs
                        if n is not self.goaln and self.T.nodes[n]["active"]
                    ],
                    key=lambda u: self.T.nodes[u]["lmc"],
                )
                return nx.subgraph(self.T, ccs).copy()

    def get_lmc(self, x):
        gx = self.T.nodes[self.goaln]["point"]
        return dotself(gx - x)

    def resample(self, new_world=None):
        # get new world
        if new_world is not None:
            self.world = new_world
        # set active nodes
        self.calculate_node_collisions()
        active = [n for n in self.T.nodes if self.T.nodes[n]["active"]]
        self.T = self.T.subgraph(active)

        # update T
        self.T = self.get_vbot_tree()

        print("resampling")
        i = 0
        closer_to_goal = False
        while not closer_to_goal:
            i += 1
            print(str(i).zfill(3), end="\r")
            x_rand = self.get_rand_pt()
            # x_nearest <- Nearest(T, x_rand)
            v_nearest = self.nearest_nodes(x_rand)[0]
            # x_new <- Steer(x_nearest, x_rand)
            x_new = x_rand
            # if ObstacleFree(x_new, x_nearest) then
            if not self.collision(v_nearest, x_new):
                # X_near <- Near(T, x_new, DX)
                V_near = self.near(x_new, self.dx)
                # V <- V U {x_new}
                new_node = uuid.uuid4()
                # x_min <- x_nearest
                v_min = v_nearest
                # TODO simplify this loop
                # connect along minimum-cost path
                c_min = self.get_cost(v_nearest, x_new)
                for v_near in V_near:
                    coll = not self.collision(v_near, x_new)
                    cost = self.get_cost(v_near, x_new)
                    if coll and cost < c_min:
                        v_min = v_near
                        c_min = cost
                # rewire the tree
                self.T.add_node(
                    new_node,
                    point=x_new,
                    cost=self.get_cost(v_min, x=x_new),
                    active=True,
                    lmc=self.get_lmc(x_new),
                )
                dist = self.ndist(new_node, v_min)
                self.T.add_edge(v_min, new_node, dist=dist, active=True)
                for v in V_near:
                    coll = not self.collision(v, new_node)
                    cost = self.get_cost(new_node)
                    cost += self.get_cost(new_node, self.T.nodes[v]["point"])
                    if coll and cost < self.get_cost(v):
                        v_parent = next(self.T.predecessors(v), None)
                        if v_parent is not None:
                            dist = self.ndist(v, new_node)
                            self.T.add_edge(new_node, v, dist=dist, active=True)
                            self.T.remove_edge(v_parent, v)
            else:
                continue
            if self.T.nodes[new_node]["lmc"] < self.T.nodes[self.closest]["lmc"]:
                self.closest = new_node
                closer_to_goal = True
                print("found new closest")
            elif i > 100:
                closer_to_goal = True
                print("exhausted tries")
            else:
                continue
        print("done! Tn={}".format(len(self.T)))


class RRTStar(RRT):
    def __init__(self, K, world, dx):
        super().__init__(K, world)
        self.dx = dx

    def get_cost(self, v, T, x=None):
        if x is None:
            return T.nodes[v]["cost"]
        else:
            dist = euclidean(T.nodes[v]["point"], x)
            return T.nodes[v]["cost"] + dist

    def build_rrt(self, start_point, end_point):
        T = nx.DiGraph()
        start_node = uuid.uuid4()
        T.add_node(start_node, point=start_point, cost=0.0, active=True)
        pts_added = set()
        for _ in tqdm(range(self.K)):
            # x_rand <- SampleFree;
            rand_idx = np.random.choice(self.free.shape[0])
            if rand_idx in pts_added:
                self.K += 1
                continue
            x_rand = self.free[rand_idx]
            # x_nearest <- Nearest(T, x_rand)
            v_nearest = self.nearest(T, x_rand)[0]
            # x_new <- Steer(x_nearest, x_rand)
            x_new = x_rand
            # if ObstacleFree(x_new, x_nearest) then
            if not self.collision_check(v_nearest, x_new, T):
                # X_near <- Near(T, x_new, DX)
                V_near = self.near(T, x_new, self.dx)
                # V <- V U {x_new}
                new_node = uuid.uuid4()

                # x_min <- x_nearest
                v_min = v_nearest
                # TODO simplify this loop
                # connect along minimum-cost path
                c_min = self.get_cost(v_nearest, T, x_new)
                for v_near in V_near:
                    coll = not self.collision_check(v_near, x_new, T)
                    cost = self.get_cost(v_near, T, x_new)
                    if coll and cost < c_min:
                        v_min = v_near
                        c_min = cost
                # rewire the tree
                dist = euclidean(x_new, T.nodes[v_min]["point"])
                T.add_node(
                    new_node,
                    point=x_new,
                    cost=self.get_cost(v_min, T, x=x_new),
                    active=True,
                )
                T.add_edge(v_min, new_node, dist=dist, active=True)
                for v_near in V_near:
                    coll = not self.collision_check(v_near, x_new, T)
                    cost = self.get_cost(new_node, T) + self.get_cost(
                        new_node, T, T.nodes[v_near]["point"]
                    )

                    if coll and cost < self.get_cost(v_near, T):
                        v_parent = next(T.predecessors(v_near), None)
                        if v_parent is not None:
                            dist = self.dist_nodes(v_near, new_node, T)
                            T.add_edge(new_node, v_near, dist=dist, active=True)
                            T.remove_edge(v_parent, v_near)

                pts_added.add(rand_idx)
            else:
                self.K += 1
                continue

        end_node = self.get_end_node(end_point, T)

        return T, start_node, end_node
'''


class RRT(object):
    def __init__(self, start, goal, k):
        self.start = start
        self.goal = goal
        self.k = k

    def nearest_nodes(self, T: nx.DiGraph, x: np.array) -> list:
        def distance(u: uuid.UUID):
            x1 = T.nodes[u]["point"]
            x2 = x
            return dotself(x1 - x2)

        return sorted([n for n in T.nodes], key=distance)

    def get_end_node(self, T, world, end_point):
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

    def dist_nodes(self, T, u, v):
        p1 = T.nodes[u]["point"]
        p2 = T.nodes[v]["point"]
        return euclidean(p1, p2)

    def collision(self, T: nx.DiGraph, world, v1, v2) -> bool:
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

    def near(self, T, p, r):
        """Get nodes within `r` of point `p`."""
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

    def path(self, T, start, end):
        node_path = nx.shortest_path(T, source=start, target=end, weight="dist")
        point_path = np.array([T.nodes[n]["point"] for n in node_path])
        return point_path


class RRTa(RRT):
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
            rand_idx = np.random.choice(free.shape[0])
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


class RRTaStar(RRT):
    def __init__(self, start, goal, k, dx):
        super().__init__(start, goal, k)
        self.dx = dx

    def get_cost(self, T, v, x=None):
        if x is None:
            return T.nodes[v]["cost"]
        else:
            dist = euclidean(T.nodes[v]["point"], x)
            return T.nodes[v]["cost"] + dist

    def make(self, world: np.array):
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
            rand_idx = np.random.choice(free.shape[0])
            x = free[rand_idx]
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
