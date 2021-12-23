import numpy as np
import numba as nb
from math import atan
from rrt import RRTstar

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

    def dist(p):
        v = p2 - p1
        return cross(v, p - p1) / r2norm(p1, p2)

    return dist


def desired_course(chi_inf, d, k=0.2):
    return -chi_inf * (2.0 / np.pi) * atan(k * d)


if __name__ == "__main__":
    from rrt import RRTstar, get_rrt_LC
    from world_gen import make_world, get_rand_start_end
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    # make world
    w, h = 256, 256
    world = np.zeros((w, h))
    xstart, xgoal = get_rand_start_end(world)

    # make RRTS
    rrts = RRTstar(world, 100)
    T = rrts.make(xstart, xgoal, 128)

    # plot RRT and world
    lc = get_rrt_LC(T)
    ax[1].add_collection(lc)

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
