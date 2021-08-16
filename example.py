from rrt import world_gen, rrt
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == "__main__":
    w, h = 128, 128
    world = world_gen.make_perlin_world(w, h, 20, thresh=0.4)
    og = world_gen.ObstacleGenerator(world)

    fig, ax = plt.subplots(
        ncols=2, nrows=2, sharex=True, sharey=True, tight_layout=True
    )
    ax_idx = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}

    ax[0, 0].imshow(world.T, origin="lower", cmap=cm.get_cmap("Greys"))
    ax[0, 0].set_title("Create Perlin Noise World (As Array)")
    ax[0, 1].set_title("Decompose Obstacles into Quad-Tree")
    og.plot_rects(ax[0, 1])
    ax[1, 0].set_title("Create RRT in Free Space")
    og.plot_rects(ax[1, 0])
    ax[1, 1].set_title("Traverse RRT to find Path from Start to Finish")
    og.plot_rects(ax[1, 1])

    start, goal = og.get_rand_start_end()
    # mark start, end on each
    for k in ax_idx.values():
        ax[k].set_aspect("equal")
        ax[k].scatter(start[0], start[1], c="r", marker="o")
        ax[k].scatter(goal[0], goal[1], c="b", marker="*")

    tree = rrt.RRT(og, 300)
    T, start_node, end_node = tree.build_rrt(start, goal)
    path = tree.path(T, start_node, end_node)

    ax[1, 1].plot(path[:, 0], path[:, 1], "r-", linewidth=2)

    pos = {}
    for n in T.nodes:
        pos[n] = T.nodes[n]["point"]

    nx.draw_networkx(
        T,
        pos,
        arrows=False,
        node_size=5,
        ax=ax[1, 0],
        with_labels=False,
        edge_color="tan",
        node_color="cadetblue",
    )
    nx.draw_networkx(
        T,
        pos,
        arrows=False,
        node_size=3,
        ax=ax[1, 1],
        with_labels=False,
        edge_color="wheat",
        node_color="lightblue",
    )
    plt.show()
