from rrtpp import world_gen
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.pyplot import Circle
import networkx as nx
from matplotlib import cm
import numpy as np
from rrtpp.rrt import RRTstandard, RRTstar, RRTinformed, get_rand_start_end
from matplotlib.colors import Normalize
from pyinstrument.profiler import Profiler
from random import randint

if __name__ == "__main__":
    world_size = (512, 512)
    N = 5000
    r_rewire = 512
    r_goal = 40

    # world = world_gen.make_world(world_size, (8, 8), thresh=0.33)
    # world = world | world_gen.make_world(world_size, (2, 2), thresh=0.35)

    #### world with walls
    # w, h = (1000, 1000)
    # world = np.zeros((h, w), dtype=bool)
    # walls = 0
    # while walls < w - 100:
    #     gap = randint(200, h - 200)
    #     walls += randint(100, 400)
    #     world[: gap - 20, walls - 2 : walls + 2] = 1
    #     world[gap + 20 :, walls - 2 : walls + 2] = 1

    # xstart, xgoal = get_rand_start_end(world)

    ####### World with a single square
    # world = np.zeros(world_size, dtype=bool)

    # h, w = world_size
    # squarey1 = int(float(h) / 2 - float(h / 6))
    # squarey2 = int(float(h) / 2 + float(h / 6))
    # squarex1 = int(float(w) / 2 - float(w / 6))
    # squarex2 = int(float(w) / 2 + float(w / 6))
    # world[squarex1:squarex2, squarey1:squarey2] = 1
    # xstart = np.array([int(h / 9), int(w / 9)])
    # xgoal = np.array([int(8 * h / 9), int(8 * w / 9)])

    h, w = world_size
    world = np.zeros((1000, 1000), dtype=bool)
    world[250:750, 400:600] = 1
    world[450:480, 400:600] = 0
    xstart, xgoal = np.array([500, 50]), np.array([500, 950])

    fig, ax = plt.subplots()
    # show image
    ax.imshow(world.T, cmap=cm.get_cmap("Greys"), origin="lower", zorder=1)
    # show start
    ax.scatter(xstart[0], xstart[1], s=75, c="g", marker="^", zorder=5)
    # show goal
    ax.scatter(xgoal[0], xgoal[1], s=75, c="r", marker="*", zorder=5)
    # show goal circle
    gls_circ = Circle(xgoal, r_goal, fill=False, zorder=2)
    ax.add_artist(gls_circ)
    # RRT Star
    rrt_star = RRTstar(world, N)
    prof = Profiler(interval=0.001)
    prof.start()
    T = rrt_star.make(xstart, xgoal, 50)
    prof.stop()
    prof.print()

    costs2 = np.array(list(nx.get_edge_attributes(T, "cost").values()))
    norm = Normalize(vmin=costs2.min(), vmax=costs2.max())
    colors = cm.get_cmap("viridis")(norm(costs2))

    def get_edges_np(T, points):
        edges = np.empty((len(T.edges), 2, 2))
        for i, (e1, e2) in enumerate(T.edges):
            edges[i, 0, :] = points[e1]
            edges[i, 1, :] = points[e2]
        return edges

    lines = get_edges_np(T, rrt_star.points)
    ln2 = LineCollection(lines, colors=colors)
    ax.add_collection(ln2)

    fig2, ax2 = plt.subplots()
    ax2.plot(range(costs2.shape[0]), norm(costs2))

    # nodes = np.array([rrt_star.T.nodes[n]["point"] for n in rrt_star.T])
    # ax.scatter(nodes[:, 0], nodes[:, 1], c="k", marker=".", zorder=1)

    # path = rrt_star.path()
    # drawpath = []
    # for j, p in enumerate(path[:-1, :]):
    #     drawpath.append((path[j, :], path[j + 1, :]))
    # pathln = LineCollection(drawpath, colors="b", lw=4)
    # ax.add_collection(pathln)

    ax.set_title("RRT Star")
    ax.axis("off")
    plt.show()
