from scipy import spatial
import numpy as np
import world_gen
import uuid, tqdm
import networkx as nx


def points_on_line(start, end, res):
    v = end - start
    # unit vector pointing from start to end
    pvec = v / np.linalg.norm(v)
    # total distance
    d = spatial.distance.euclidean(start, end)
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
        self.dx = 10

    def near(self, T, x_rand, dx):
        dx2 = dx * dx

        def near(v):
            return dotself(x_rand - T.nodes[v]["point"])

        return [v for v in T.nodes if near(v) < dx2]

    def build_rrt(self, start_point, end_point):
        T = nx.Graph()
        start_node = uuid.uuid4()
        T.add_node(start_node, point=start_point)
        pts_added = set()
        for _ in tqdm(range(self.K)):
            # random free point
            rand_idx = np.random.choice(self.free.shape[0])
            if rand_idx in pts_added:
                self.K += 1
                continue
            else:
                pts_added.add(rand_idx)
            x_rand = self.free[rand_idx]
            v_nearest = self.obstacle_free(T, x_rand)
            if not v_nearest:
                self.K += 1
                continue

            new_node = uuid.uuid4()
            T.add_node(new_node, point=x_rand)
            dist = spatial.distance.euclidean(x_rand, T.nodes[v_nearest]["point"])
            T.add_edge(v_nearest, new_node, dist=dist)
        v_nearest = self.obstacle_free(T, end_point)
        if not v_nearest:
            v_nearest = self.nearest_vertices(end_point, T)[0]
            print("Could not path to goal, finding closest.")
        end_node = uuid.uuid4()
        T.add_node(end_node, point=end_point)
        dist = spatial.distance.euclidean(end_point, T.nodes[v_nearest]["point"])
        T.add_edge(v_nearest, end_node, dist=dist)
        return T, start_node, end_node

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

    def nearest_vertices(self, x_rand, T):
        def distance(u):
            x1 = T.nodes[u]["point"]
            x2 = x_rand
            return spatial.distance.euclidean(x1, x2)

        return sorted([n for n in T.nodes], key=distance)

    def collision_check(self, v, x_rand, T):
        for p in points_on_line(T.nodes[v]["point"], x_rand, res=0.5):
            if self.rtree.count(p) != 0:
                return True
        return False
