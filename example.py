from rrt import grow
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

if __name__ == "__main__":
    w, h = 128, 128
    world = np.squeeze(grow.make_perlin_world((1, 128, 128), (1, 4, 4), 20, thresh=0.4))
    start, goal = grow.get_rand_start_end(world)

    rrt = grow.RRTGrow(w, h, start, goal, 1000, 16)
    rrt.make()

    pos = grow.make_pos(rrt.T)

    fig, ax = plt.subplots()
    ax.imshow(world.T, origin="lower", cmap=cm.get_cmap("Greys"))
    print(rrt.T)
    nx.draw_networkx(
        rrt.T,
        pos,
        arrows=True,
        node_size=5,
        ax=ax,
        with_labels=False,
        edge_color="tan",
        node_color="cadetblue",
    )
    ax.autoscale()
    plt.show()
