from matplotlib.pyplot import delaxes
from math import floor
from networkx.algorithms.shortest_paths.unweighted import predecessor
from networkx.generators.geometric import euclidean
from scipy.spatial.distance import euclidean
import numpy as np
from . import world_gen
import uuid
from tqdm import tqdm
import networkx as nx
from collections import deque

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


def dotself(u):
    return np.dot(u, u)


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
                v_nearest, end_node, dist=self.dist_nodes(end_node, v_nearest, T)
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
                    v_nearest, new_node, dist=self.dist_nodes(v_nearest, new_node, T)
                )
            pts_added.add(rand_idx)

        end_node = self.get_end_node(end_point, T)

        return T, start_node, end_node


class RRTx(RRT):
    def __init__(self, K, world):
        super().__init__(K, world)
        self.T = nx.DiGraph()

        self.gamma = 10.0
        self.delta = 1.0
        self.zeta = 1.0

        self.Q = deque()

    # def update_rrt(self, new_world):
    #     # update world
    #     self.world = new_world
    #     # go through edges and find collisions
    #     for e1, e2 in self.T.edges:
    #         if self.collision_check(e1, e2, self.T):
    #             self.T[e1][e2]["active"] = False
    #             self.T.nodes[e1]["active"] = False
    #             self.T.nodes[e2]["active"] = False
    #         else:
    #             self.T[e1][e2]["active"] = True
    #             self.T.nodes[e1]["active"] = True
    #             self.T.nodes[e2]["active"] = True

    def shrinking_ball_radius(self):
        d1 = self.gamma * np.log10(len(self.T.nodes))
        d2 = self.zeta * len(self.T.nodes)
        return min((d1/d2)**(1/self.D), self.delta)

    def reduce_inconsistency(self):
        while len(self.Q) > 0 and 

    def rrtx(self, start, end):
        # V <- v_goal
        self.end_node_id = uuid.uuid4()
        self.v_goal = self.end_node_id
        self.T.add_node(self.end_node_id, active=True, point=end)
        # v_bot <- v_start
        self.start_id = uuid.uuid4()
        self.v_bot = self.start_id
        self.T.add_node(self.start_id, active=True, point=start)
        while self.v_bot != self.v_goal:
            r = self.shrinking_ball_radius()

        #   r <- shrinking_ball_radius
        #   if obstacles_have_changed:
        #       update_obstacles
        #   if robot is moving then
        #       v_bot <- update_robot(v_bot)
        #   v <- random_node
        #   v_nearest <- nearest(v)
        #   if d(v, v_nearest) > delta then
        #       v <- saturate(v, v_nearest)
        #   if v not in X_obs then
        #       extend(v, r)
        #   if v in V then
        #       rewire neighbors(v);
        #       reduce inconsistency();


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
