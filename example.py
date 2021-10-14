from rrtpp import world_gen
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.pyplot import Circle
import networkx as nx
from matplotlib import cm
import numpy as np
from rrtpp.rrt import RRTStandard, RRTStar, RRTStarInformed


if __name__ == "__main__":
    world_size = (64, 64)
    N = 90
    r_rewire = 32
    r_goal = 10
    cmapstr = "viridis"

    world = world_gen.make_world(world_size, (4, 4), thresh=0.33)

    rrt_standard = RRTStandard(world)
    xstart, xgoal = rrt_standard.get_rand_start_end()

    fig, axs = plt.subplots(ncols=3)
    for ax in axs:
        # show image
        ax.imshow(world.T, cmap=cm.get_cmap("Greys"), origin="lower", zorder=1)
        # show start
        ax.scatter(xstart[0], xstart[1], s=75, c="g", marker="^", zorder=5)
        # show goal
        ax.scatter(xgoal[0], xgoal[1], s=75, c="r", marker="*", zorder=5)
        # show goal circle
        gls_circ = Circle(xgoal, r_goal, fill=False, zorder=2)
        ax.add_artist(gls_circ)

    def get_edges_np(T):
        return np.array(
            [(T.nodes[e1]["point"], T.nodes[e2]["point"]) for (e1, e2) in T.edges]
        )

    # RRT Standard
    vstart1, vend1 = rrt_standard.make(xstart, xgoal, N)

    costs1 = np.array(list(nx.get_edge_attributes(rrt_standard.T, "cost").values()))
    costs1 /= costs1.max() - costs1.min()
    cmap = cm.get_cmap(cmapstr)
    ln1 = LineCollection(get_edges_np(rrt_standard.T), colors=cmap(costs1))
    axs[0].add_collection(ln1)

    # RRT Star
    rrt_star = RRTStar(world)
    vstart2, vend2 = rrt_star.make(xstart, xgoal, N, r_rewire)

    costs2 = np.array(list(nx.get_edge_attributes(rrt_star.T, "cost").values()))
    costs2 /= costs2.max() - costs2.min()
    cmap = cm.get_cmap(cmapstr)
    ln2 = LineCollection(get_edges_np(rrt_star.T), colors=cmap(costs2))
    axs[1].add_collection(ln2)

    rrt_stari = RRTStarInformed(world)
    vstart3, vend3 = rrt_stari.make(xstart, xgoal, N, r_rewire, r_goal)

    costs3 = np.array(list(nx.get_edge_attributes(rrt_stari.T, "cost").values()))
    costs3 /= costs3.max() - costs3.min()
    cmap = cm.get_cmap(cmapstr)
    ln3 = LineCollection(get_edges_np(rrt_stari.T), colors=cmap(costs3))
    axs[2].add_collection(ln3)

    ell = rrt_stari.data["ellipse"]
    ell.zorder = 5
    axs[2].add_artist(ell)

    for ax, rrts in zip(axs, (rrt_standard, rrt_star, rrt_stari)):
        nodes = np.array([rrts.T.nodes[n]["point"] for n in rrts.T])
        ax.scatter(nodes[:, 0], nodes[:, 1], c="k", marker=".", zorder=1)

        path = rrts.path()
        drawpath = []
        for j, p in enumerate(path[:-1, :]):
            drawpath.append((path[j, :], path[j + 1, :]))
        pathln = LineCollection(drawpath, colors="b", lw=4)
        ax.add_collection(pathln)

    axs[0].set_title("RRT (Standard)")
    axs[1].set_title("RRT Star")
    axs[2].set_title("RRT Star Informed")

    for a in axs:
        a.axis("off")

    plt.show()
