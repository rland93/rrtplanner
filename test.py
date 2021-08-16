from re import L
from networkx.algorithms.assortativity import mixing
from networkx.classes.function import number_of_nodes
import world_gen
import matplotlib.pyplot as plt
from matplotlib import cm
import networkx as nx
import numpy as np
from scipy import spatial
import rtree, uuid
from tqdm import tqdm
import os
import multiprocessing as mp


class RRT(object):
    def __init__(self, og: world_gen.ObstacleGenerator, x_init, x_goal, K):
        self.og = og
        self.K = K
        self.T = nx.DiGraph()
        self.T.add_node(uuid.uuid4(), point=x_init + np.array((0.5, 0.5)))
        self.free_spaces = og.free_space_list + np.array((0.5, 0.5))

    def new_conf(self, x_near, x_rand):
        """configured for holonomic problem..."""
        return x_rand

    def build_rrt(self):
        for _ in range(self.K):
            rand_free_idx = np.random.choice(self.free_spaces.shape[0])
            x_rand = self.free_spaces[rand_free_idx]

            x_near = self.nearest_vertex(x_rand)
            if x_near is None:
                continue
            x_new = self.new_conf(self.T.nodes[x_near]["point"], x_rand)
            new_node = uuid.uuid4()
            self.T.add_node(new_node, point=x_new)
            self.T.add_edge(x_near, new_node)

    def nearest_vertex(self, x_rand):
        def distance(u):
            x1 = self.T.nodes[u]["point"]
            x2 = x_rand
            return spatial.distance.euclidean(x1, x2)

        distn = sorted([n for n in self.T.nodes], key=distance)

        for n in distn:
            if self.collision_check(n, x_rand):
                continue
            else:
                print("found", n)
                return n

    def collision_check(self, n, x_rand):
        n_ = self.T.nodes[n]["point"]
        d2_final = np.dot(x_rand - n_, x_rand - n_)
        pvec = (x_rand - n_) / np.sqrt(d2_final)
        d2 = 0.5
        while d2 < d2_final:
            n_ += pvec
            d2 = np.dot(x_rand - n_, x_rand - n_)
            coll_generator = self.og.obs_rtree.intersection(tuple(n_))
            if next(coll_generator, None) is None:
                continue
            else:
                return True
        return False


if __name__ == "__main__":

    w, h = 128, 128

    world = world_gen.make_perlin_world(w, h, 5, thresh=0.4)
    og = world_gen.ObstacleGenerator(world)

    start, end = og.get_rand_start_end()

    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(world.T, origin="lower", cmap=cm.get_cmap("PuRd"))
    og.plot_rects(ax[1])
    ax[1].scatter(start, end)

    x_init, x_goal = og.get_rand_start_end()

    rrt = RRT(og, x_init, x_goal, 250)
    rrt.build_rrt()

    pos = {}
    for n in rrt.T.nodes:
        pos[n] = rrt.T.nodes[n]["point"]

    nx.draw_networkx(rrt.T, pos, arrows=False, node_size=5, ax=ax[1], with_labels=False)

    plt.show()
