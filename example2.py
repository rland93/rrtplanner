from rrt import world_gen, rrt
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

if __name__ == "__main__":

    w, h = 128, 128
    world = world_gen.make_perlin_world(w, h, 20, seed=92103, thresh=0.4)
    og = world_gen.ObstacleGenerator(world)
    start, goal = og.get_rand_start_end()

    fig, ax = plt.subplots(ncols=2, figsize=(8, 8))
    for a in ax:
        og.plot_rects(a)
        a.scatter(start[0], start[1], marker="o", color="b")
        a.scatter(goal[0], goal[1], marker="*", color="r")

    rrt_r = rrt.RRTStandard(og, 500)
    T_r, start_node_r, end_node_r = rrt_r.build_rrt(start, goal)
    path_r = rrt_r.path(T_r, start_node_r, end_node_r)

    rrt_s = rrt.RRTStar(og, 500, 24)
    T_s, start_node_s, end_node_s = rrt_s.build_rrt(start, goal)
    path_s = rrt_s.path(T_s, start_node_s, end_node_s)

    for a, T, path in zip(ax, (T_r, T_s), (path_r, path_s)):
        a.plot(path[:, 0], path[:, 1], color="green", linestyle="--", linewidth=2)
        pos = {}
        for n in T.nodes:
            pos[n] = T.nodes[n]["point"]

        nx.draw_networkx(
            T,
            pos,
            arrows=True,
            node_size=5,
            ax=a,
            with_labels=False,
            edge_color="tan",
            node_color="cadetblue",
        )
    plt.show()
