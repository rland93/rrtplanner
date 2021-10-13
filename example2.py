from rrtpp import world_gen, rrt
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Ellipse
from matplotlib import cm
import numpy as np
import networkx as nx
from matplotlib.colors import to_rgba


if __name__ == "__main__":
    world_size = (256, 256)
    n = 300
    dx = 128
    rd = 16

    world = world_gen.make_world(world_size, (4, 4), thresh=0.43)

    # world = np.zeros(shape=world_size, dtype=bool)
    start, goal = rrt.get_rand_start_end(world, bias=False)
    fig, ax = plt.subplots()

    T, vstart, vend, ell = rrt.make_RRT_starinformed(world, start, goal, n, dx, rd)

    ax.add_artist(ell)
    ax.imshow(world.T, cmap=cm.get_cmap("Greys"), origin="lower")

    costs = np.array(list(nx.get_edge_attributes(T, "cost").values()))
    costs /= costs.max() - costs.min()
    cmap = cm.get_cmap("viridis")
    pos = nx.get_node_attributes(T, "point")

    edges = np.array(
        [(T.nodes[e1]["point"], T.nodes[e2]["point"]) for (e1, e2) in T.edges]
    )
    eln = LineCollection(edges, colors=cmap(costs))
    ax.add_collection(eln)
    path = rrt.path(T, vstart, vend)
    path = np.array([path[:-1], path[1:]])
    path = np.moveaxis(path, 0, 1)

    ln = LineCollection(path, colors="red", linewidths=2.5)
    ax.add_collection(ln)

    nodes = np.array([T.nodes[n]["point"] for n in T])

    ax.scatter(nodes[:, 0], nodes[:, 1], c="k", marker=".", zorder=1)
    ax.scatter(start[0], start[1], s=100, c="g", marker="^", zorder=10)
    ax.scatter(goal[0], goal[1], s=100, c="r", marker="*", zorder=10)

    gls_circ = Circle(goal, rd, fill=False)
    ax.add_artist(gls_circ)

    plt.show()
