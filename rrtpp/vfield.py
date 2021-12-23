import numpy as np
import numba as nb
from math import atan
from rrt import RRTstar
from scipy import spatial
import networkx as nx
import functools
import pyvoronoi

PI2 = np.pi / 2.0


@nb.njit(fastmath=True)
def r2norm(p1, p2=None):
    if p2 is not None:
        v = p2 - p1
        return np.sqrt(v[0] * v[0] + v[1] * v[1])
    else:
        return np.sqrt(p1[0] * p1[0] + p1[1] * p1[1])


def sign(x):
    if x > 0:
        return 1.0
    else:
        return -1.0


@nb.njit(fastmath=True)
def cross(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]


def dist2line(p1, p2) -> callable:
    """get function that calculates distance to line between p1 and p2"""
    if all(p1 == p2):
        raise ValueError("p1 and p2 must not be equal")

    def dist(p):
        v = p2 - p1
        return cross(v, p - p1) / r2norm(p1, p2)

    return dist


def desired_course(chi_inf, d, k=0.2):
    return -chi_inf * (2.0 / np.pi) * atan(k * d)


if __name__ == "__main__":
    from matplotlib import cm
    from matplotlib.collections import LineCollection
    from rrt import RRTstar, get_rrt_LC
    from world_gen import make_world, get_rand_start_end
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    # make world
    w, h = 64, 64
    world = make_world((w, h), (16, 16))
    world = world | make_world((w, h), (4, 4))
    world = world | make_world((w, h), (2, 2))
    xstart, xgoal = get_rand_start_end(world)
    world = np.zeros((w, h), dtype=int)

    # make RRTS
    rrts = RRTstar(world, 20)
    T, gv = rrts.make(xstart, xgoal, 256)

    # plot RRT and world
    lc = get_rrt_LC(T)
    ax[1].add_collection(lc)
    for a in ax:
        a.scatter(xstart[0], xstart[1], c="r", s=50, label="start", marker="o")
        a.scatter(xgoal[0], xgoal[1], c="b", s=50, label="goal", marker="*")
        a.imshow(world.T, cmap="Greys", origin="lower")
        a.set_xlim(0, w)
        a.set_ylim(0, h)

    # get voronoi regions
    path = nx.shortest_path(T, 0, gv, weight="dist")
    pv = pyvoronoi.Pyvoronoi(len(path) * 5)
    pathlc = []
    # identify paths that are in the center box
    segmask = []
    for i in range(len(path) - 1):
        p1 = T.nodes[path[i]]["pos"]
        p2 = T.nodes[path[i + 1]]["pos"]
        pv.AddSegment([p1, p2])
        segmask.append(True)
        pathlc.append([p1, p2])
        trans = [0, w, 0, h]
        rot1 = [0, 0, 1, 1]
        rot2 = [1, 1, 0, 0]
        for t, r1, r2 in zip(trans, rot1, rot2):
            # it's getting late
            a, b = [None, None], [None, None]
            a[r1] = p1[r1] + ((t - p1) * 2)[r1]
            b[r1] = p2[r1] + ((t - p2) * 2)[r1]
            a[r2] = p1[r2]
            b[r2] = p2[r2]
            pv.AddSegment([a, b])
            segmask.append(False)
    segmask = np.array(segmask)

    # plot voronoi
    pv.Construct()
    cells = pv.GetCells()
    edges = pv.GetEdges()
    vertices = pv.GetVertices()

    lcs = []
    for i, c in enumerate(cells):
        cell_lines = []
        if segmask[c.site]:
            for edge in c.edges:
                v1 = edges[edge].start
                v2 = edges[edge].end
                if v1 != -1:
                    p1 = vertices[v1]
                    p1 = np.array((p1.X, p1.Y))
                else:
                    p1 = None
                if v2 != -1:
                    p2 = vertices[v2]
                    p2 = np.array((p2.X, p2.Y))
                else:
                    p2 = None

                if p1 is not None and p2 is not None:
                    cell_lines.append((p1, p2))
            lcs.append(LineCollection(cell_lines, color="green", alpha=0.5))

    for lc in lcs:
        ax[0].add_collection(lc)
    ax[0].add_collection(LineCollection(pathlc, color="red"))

    ax[0].set_xlim(-w * 0.1, w * 1.1)
    ax[0].set_ylim(-h * 0.1, h * 1.1)

    plt.show()

    """
    # all edge points
    epoints = np.empty((len(T.edges), 2, 2))
    for i, (e1, e2) in enumerate(T.edges):
        p1, p2 = rrts.points[e1], rrts.points[e2]
        epoints[i, 0, :] = p1
        epoints[i, 1, :] = p2

    angles = np.empty((w, h))
    points = np.empty((w, h, 2))
    # for each point, find closest edge and calculate angle
    for p in tqdm(np.ndindex((w, h)), total=w * h):
        # find closest edge
        dists = np.array(
            [
                dist2line(epoints[i, 0, :], epoints[i, 1, :])(p)
                for i in range(len(T.edges))
            ]
        )
        i = np.argmin(dists)
        chi = np.arctan2(
            epoints[i, 1, 1] - epoints[i, 0, 1], epoints[i, 1, 0] - epoints[i, 0, 0]
        )
        d = dists[i]

        chi_inf = PI2
        course = desired_course(chi_inf, d)
        angles[p] = course + chi
        points[p] = p

    ax[0].quiver(
        points[:, :, 1],
        points[:, :, 0],
        np.sin(angles),
        np.cos(angles),
        scale=1.0,
        width=0.1,
        units="xy",
    )
    """
    plt.show()

    # import matplotlib.pyplot as plt
    # from matplotlib import cm
    # from random import uniform

    # w, h = 80, 80

    # for ax, i in zip(plt.subplots(1, 2, figsize=(8, 4))[1], range(2)):
    #     if i == 0:
    #         p1 = np.array((40, 80))
    #         p2 = np.array((40, 0))
    #     elif i == 1:
    #         p1 = np.array((40, 0))
    #         p2 = np.array((40, 80))

    #     # heading when on the line
    #     chi = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    #     # function for dist to line
    #     f = dist2line(p1, p2)

    #     dists = np.empty((w, h))
    #     angles = np.empty((w, h))
    #     points = np.empty((w, h, 2))

    #     for p in np.ndindex((w, h)):
    #         y = f(p)
    #         chi_inf = PI2
    #         dists[p] = abs(y)
    #         course = desired_course(chi_inf, y)
    #         angles[p] = course + chi
    #         points[p] = p

    #     ax.imshow(dists, cmap=cm.get_cmap("Blues"), origin="lower")
    #     ax.quiver(
    #         points[:, :, 1],
    #         points[:, :, 0],
    #         np.sin(angles),
    #         np.cos(angles),
    #         scale=1.0,
    #         width=0.1,
    #         units="xy",
    #     )
    # plt.show()
