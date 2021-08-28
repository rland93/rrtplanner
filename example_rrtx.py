from rrt import world_gen, rrt
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib import animation
from matplotlib.collections import LineCollection
import copy

if __name__ == "__main__":
    tworld = world_gen.make_perlin_world(
        (256, 128, 128), (4, 4, 4), 5, seed=92103, thresh=0.4
    )

    def animate(worlds, Ts, poss):
        print(worlds.shape[0], len(Ts), len(poss))
        assert worlds.shape[0] == len(Ts)
        assert worlds.shape[0] == len(poss)
        fig, ax = plt.subplots(figsize=(10, 10))

        sc = plt.scatter([], [])
        ln = LineCollection([])
        ax.add_collection(ln)

        def anim(i):
            T = Ts[i]
            nodes = np.array(
                [T.nodes[n]["point"] for n in T if T.nodes[n]["active"] == True]
            )
            edges = [
                (T.nodes[e1]["point"], T.nodes[e2]["point"])
                for (e1, e2) in T.edges
                if T[e1][e2]["active"] == True
            ]
            sc.set_offsets(nodes)
            ln.set_segments(edges)

            im = ax.imshow(
                worlds[i, :, :].T,
                origin="lower",
                cmap=cm.get_cmap("Greys"),
                animated=True,
            )
            return (im, sc, ln)

        frames = range(worlds.shape[0])

        return animation.FuncAnimation(fig, anim, frames=frames, interval=50, blit=True)

    rrt = rrt.RRTx(800, tworld[0])
    start, goal = rrt.get_rand_start_end()

    T, start, end = rrt.build_rrt(start, goal)
    pos = {}
    for n in T.nodes:
        pos[n] = T.nodes[n]["point"]

    Ts = []
    poss = []

    for new_world in tworld:
        rrt.update_rrt(new_world)
        Ts.append(copy.deepcopy(rrt.T))
        pos = {}
        for n in T.nodes:
            pos[n] = T.nodes[n]["point"]
        poss.append(pos)

    anim = animate(tworld, Ts, poss)
    plt.show()
