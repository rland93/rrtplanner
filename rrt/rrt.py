from networkx.algorithms.bipartite.matching import INFINITY
from networkx.algorithms.components.weakly_connected import (
    number_weakly_connected_components,
)
from networkx.algorithms.shortest_paths.unweighted import predecessor
from networkx.generators.geometric import euclidean
from scipy.spatial.distance import euclidean
import numpy as np
from . import world_gen
import uuid
from tqdm import tqdm
import networkx as nx


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
    def __init__(self, og: world_gen.ObstacleGenerator, K):
        self.og = og
        self.K = K
        self.free = np.argwhere(og.superworld == 0) + np.array((0.5, 0.5))
        self.rtree = og.get_rtree(og.obstacles)

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

        return min([n for n in T.nodes], key=distance)

    def collision_check(self, v, x_rand, T):
        for p in points_on_line(T.nodes[v]["point"], x_rand, res=0.5):
            if self.rtree.count(p) != 0:
                return True
        return False

    def near(self, T, x_rand, dx):
        dx2 = dx * dx
        return [v for v in T.nodes if dotself(x_rand - T.nodes[v]["point"]) < dx2]

    def dist_nodes(self, u, v, T):
        p1 = T.nodes[u]["point"]
        p2 = T.nodes[v]["point"]
        return euclidean(p1, p2)

    def get_end_node(self, end_point, T):
        v_nearest = self.nearest(T, end_point)
        if not self.collision_check(v_nearest, end_point, T):
            end_node = uuid.uuid4()
            T.add_node(end_node, point=end_point)
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
    def __init__(self, og: world_gen.ObstacleGenerator, K):
        super().__init__(og, K)

    def build_rrt(self, start_point, end_point):
        T = nx.DiGraph()
        start_node = uuid.uuid4()
        T.add_node(start_node, point=start_point)
        pts_added = set()
        for _ in tqdm(range(self.K)):
            rand_idx = np.random.choice(self.free.shape[0])
            if rand_idx in pts_added:
                self.K += 1
                continue
            x_rand = self.free[rand_idx]
            x_new = x_rand
            v_nearest = self.nearest(T, x_new)
            if not self.collision_check(v_nearest, x_new, T):
                new_node = uuid.uuid4()
                T.add_node(new_node, point=x_new)
                T.add_edge(
                    v_nearest, new_node, dist=self.dist_nodes(v_nearest, new_node, T)
                )
            pts_added.add(rand_idx)

        end_node = self.get_end_node(end_point, T)

        return T, start_node, end_node


class RRTStar(RRT):
    def __init__(self, og: world_gen.ObstacleGenerator, K, dx):
        super().__init__(og, K)
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
        T.add_node(start_node, point=start_point, cost=0.0)
        pts_added = set()
        for _ in tqdm(range(self.K)):
            # x_rand <- SampleFree;
            rand_idx = np.random.choice(self.free.shape[0])
            if rand_idx in pts_added:
                self.K += 1
                continue
            x_rand = self.free[rand_idx]
            # x_nearest <- Nearest(T, x_rand)
            v_nearest = self.nearest(T, x_rand)
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
                T.add_node(new_node, point=x_new, cost=self.get_cost(v_min, T, x=x_new))
                T.add_edge(v_min, new_node, dist=dist)
                for v_near in V_near:
                    coll = not self.collision_check(v_near, x_new, T)
                    cost = self.get_cost(new_node, T) + self.get_cost(
                        new_node, T, T.nodes[v_near]["point"]
                    )

                    if coll and cost < self.get_cost(v_near, T):
                        v_parent = next(T.predecessors(v_near), None)
                        if v_parent is not None:
                            dist = self.dist_nodes(v_near, new_node, T)
                            T.add_edge(new_node, v_near, dist=dist)
                            T.remove_edge(v_parent, v_near)

                pts_added.add(rand_idx)
            else:
                self.K += 1
                continue

        end_node = self.get_end_node(end_point, T)

        return T, start_node, end_node
