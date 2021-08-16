import world_gen
import matplotlib.pyplot as plt
from matplotlib import cm
import networkx as nx
import numpy as np
from scipy import spatial
import uuid
from tqdm import tqdm


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

    def build_rrt(self, start_point):
        T = nx.DiGraph()
        T.add_node(uuid.uuid4(), point=x_init + np.array((0.5, 0.5)))
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
            x_nearest = self.obstacle_free(T, x_rand)
            if not x_nearest:
                self.K += 1
                continue

            new_node = uuid.uuid4()
            T.add_node(new_node, point=x_rand)
            T.add_edge(x_nearest, new_node)
        return T

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


if __name__ == "__main__":
    w, h = 128, 128

    world = world_gen.make_perlin_world(w, h, 20, thresh=0.4)
    og = world_gen.ObstacleGenerator(world)

    start, end = og.get_rand_start_end()

    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(world.T, origin="lower", cmap=cm.get_cmap("Greys"))
    og.plot_rects(ax[1])

    ax[1].scatter(start[0], start[1], c="r", marker="o")
    ax[1].scatter(end[0], end[1], c="b", marker="*")

    x_init, x_goal = og.get_rand_start_end()

    rrt = RRT(og, 1000)

    T = rrt.build_rrt(x_init)

    pos = {}
    for n in T.nodes:
        pos[n] = T.nodes[n]["point"]

    nx.draw_networkx(
        T,
        pos,
        arrows=False,
        node_size=5,
        ax=ax[1],
        with_labels=False,
        edge_color="tan",
        node_color="cadetblue",
    )
    ax[1].axis("equal")
    plt.show()
