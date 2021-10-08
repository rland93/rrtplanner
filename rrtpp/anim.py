import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.pyplot import Circle
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.spatial.distance import euclidean
from matplotlib import cm

from rrtpp import world_gen
from rrtpp import rrt


def animate(worlds, Ts, poss, paths, positions, goals, goal_rad, start, end):
    print(worlds.shape[0], len(Ts), len(poss))
    assert worlds.shape[0] == len(Ts)
    assert worlds.shape[0] == len(poss)
    fig, ax = plt.subplots(figsize=(16, 9))
    # sc = ax.scatter([], [], marker=".")
    ln = LineCollection([], colors="silver")
    im = ax.imshow(worlds[0, :, :].T, cmap=cm.get_cmap("Greys"), origin="lower")
    ax.add_collection(ln)
    ax.set_ylim(0, worlds.shape[1])
    ax.set_xlim(0, worlds.shape[2])
    pos = ax.scatter(*positions[0], marker=">", c="g")
    gls = ax.scatter(*goals[0], marker="*", c="r")
    gls_circ = Circle(goals[0], goal_rad, fill=False)
    ax.add_artist(gls_circ)
    pn = LineCollection([], colors="red")
    ax.add_collection(pn)

    def anim(i):
        T = Ts[i]
        nodes = np.array(
            [T.nodes[n]["point"] for n in T if T.nodes[n]["active"] == True]
        )
        edges = np.array(
            [
                (T.nodes[e1]["point"], T.nodes[e2]["point"])
                for (e1, e2) in T.edges
                if T[e1][e2]["active"] == True
            ]
        )
        path = np.array([paths[i][:-1], paths[i][1:]])
        path = np.moveaxis(path, 0, 1)
        # sc.set_offsets(nodes)
        ln.set_segments(edges)
        pn.set_segments(path)
        pos.set_offsets(positions[i])
        gls.set_offsets(goals[i])
        gls_circ.center = goals[i]
        im.set_array(worlds[i, :, :].T)
        return (im, ln, pn, pos, gls_circ)

    frames = list(range(worlds.shape[0]))

    return animation.FuncAnimation(fig, anim, frames=frames, interval=75, blit=False)


def make_animation(
    world_size: tuple,
    frames: int,
    vspeed: float,
    goal_rad: float,
    worldspeed: int,
    k: int,
    dx: int,
    movie_file: str = "./animation.mp4",
):
    worldmovespeed = max(int(frames / 32) * worldspeed, 1)
    tworld = world_gen.make_perlin_world(
        (frames, *world_size), (worldmovespeed, 4, 4), 5, thresh=0.37
    )
    start, goal = rrt.get_rand_start_end(tworld[0])
    rrta = rrt.RRTstar(start, goal, k, dx)

    Ts = []
    poss = []
    paths = []
    positions = []
    goals = []

    for world in tworld:
        goals.append(rrta.goal)
        T, startn, goaln = rrta.make(world)
        Ts.append(T)
        poss.append(rrt.make_pos(T))
        path = rrta.path(T, startn, goaln)
        paths.append(path)
        if len(path) > 1:
            diff = np.linalg.norm(path[1] - path[0])
            if diff > 0.000001:
                v = vspeed * (path[1] - path[0]) / diff
                s_new = np.array(np.floor(path[0] + v), dtype=np.int64)
                print(path[0], s_new)
                rrta.set_start(s_new)
        positions.append(path[0])
        d_to_goal = euclidean(rrta.start, rrta.goal)
        if d_to_goal < goal_rad:
            _, goal = rrt.get_rand_start_end(world, bias=False)
            rrta.set_goal(goal)

    anim = animate(tworld, Ts, poss, paths, positions, goals, goal_rad, start, goal)
    plt.show()
    anim.save(movie_file)
