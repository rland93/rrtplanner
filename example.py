from rrtpp import world_gen, rrt
from rrtpp.rrt import RRTstarInformed
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.pyplot import Circle
from matplotlib import cm
import numpy as np


if __name__ == "__main__":
    world_size = (400, 512)
    k = 300
    dx = 64
    rd = 24

    world = world_gen.make_perlin_world((1, *world_size), (1, 4, 4), 5, thresh=0.43)[
        0, :, :
    ]
    start, goal = rrt.get_rand_start_end(world, bias=False)
    rrta = rrt.RRTstarInformed()
    T = rrta.make(world, start, goal, k, dx, rd)
    fig, ax = plt.subplots()
    ax.imshow(world.T, cmap=cm.get_cmap("Greys"), origin="lower")

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
