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
        (12, 128, 128), (4, 4, 4), 5, seed=92103, thresh=0.4
    )

    def animate(worlds, Ts, poss):
        print(worlds.shape[0], len(Ts), len(poss))
        assert worlds.shape[0] == len(Ts)
        assert worlds.shape[0] == len(poss)
        fig, ax = plt.subplots(figsize=(6, 6))

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

    start, goal = rrt.get_rand_start_end(tworld[0])
    rrta = rrt.RRTStar_Adaptive(800, tworld[0], 10, start, goal)

    Ts = []
    poss = []

    for world in tworld:
        rrta.resample(new_world=world)
        Ts.append(copy.deepcopy(rrta.T))
        poss.append(rrta.get_pos())
        rrta.update_bot_pos()

    anim = animate(tworld, Ts, poss)
    plt.show()
