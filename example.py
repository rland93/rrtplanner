from rrtpp import world_gen, rrt
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.pyplot import Circle
from networkx import draw_networkx_edges
from matplotlib import cm
import numpy as np


if __name__ == "__main__":
    world_size = (512, 512)
    n = 900
    dx = 64
    rd = 20

    world = world_gen.make_world(world_size, (4, 4), thresh=0.43)
    start, goal = rrt.get_rand_start_end(world, bias=False)

    fig, axs = plt.subplots(ncols=3)

    T1, vstart1, vend1 = rrt.make_RRT_standard(world, start, goal, n)
    T2, vstart2, vend2 = rrt.make_RRT_star(world, start, goal, n, dx)
    T3, vstart3, vend3 = rrt.make_RRT_starinformed(world, start, goal, n, dx, rd)

    rrts = ((T1, vstart1, vend1), (T2, vstart2, vend2), (T3, vstart2, vend2))

    for (T, vstart, vend), ax in zip(rrts, axs):
        ax.imshow(world.T, cmap=cm.get_cmap("Greys"), origin="lower")

        # draw_networkx_edges(
        #     T, rrt.make_pos(T), ax=ax, arrowsize=6, edge_color="silver", node_size=1
        # )

        edges = np.array(
            [(T.nodes[e1]["point"], T.nodes[e2]["point"]) for (e1, e2) in T.edges]
        )
        ln = LineCollection(edges, colors="silver")
        ax.add_collection(ln)
        nodes = np.array([T.nodes[n]["point"] for n in T])

        ax.scatter(nodes[:, 0], nodes[:, 1], c="k", marker=".", zorder=1)
        ax.scatter(start[0], start[1], s=10, c="g", marker="^", zorder=10)
        ax.scatter(goal[0], goal[1], s=10, c="r", marker="*", zorder=10)

        gls_circ = Circle(goal, rd, fill=False)
        ax.add_artist(gls_circ)

    plt.show()
