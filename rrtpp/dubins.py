import matplotlib.pyplot as plt
from math import cos, sin, sqrt, atan2, acos, asin, tan
import numpy as np
from rrt import norm
import numba
from enum import Enum


from matplotlib.pyplot import arrow


class side(Enum):
    R = 1
    S = 0
    L = -1


HANDED_IDX = {
    0: (1.0, 1.0),  # right, right
    1: (1.0, -1.0),  # left, left
    2: (-1.0, 1.0),  # right, left
    3: (-1.0, -1.0),  # left, right
}


PI2 = np.pi / 2
P2I = np.pi * 2
PI = np.pi


@numba.njit(fastmath=True)
def r2norm(v):
    return sqrt(v[0] * v[0] + v[1] * v[1])


def get_circs_from_points(r, u, v):
    """get circles that are left or right of the vectors u, v."""
    # theta 1, 2
    for w in (u, v):
        # R, L
        circr = w[:2] + r * np.array((cos(w[2] - PI2), sin(w[2] - PI2)))
        circl = w[:2] + r * np.array((cos(w[2] + PI2), sin(w[2] + PI2)))
        for c in circr, circl:
            yield c


def get_tans(r, c1, c2, i):
    """get tangent points for 2 circles of radius r centered
    at c1, c2. Lines connecting these tangent points connect the
    circles."""
    d = r2norm(c2 - c1)
    rl1, rl2 = HANDED_IDX[i]
    c = (r - rl1 * r) / d
    if c * c > 1.0:
        return None
    v = (c2 - c1) / d
    h = sqrt(max(0, 1.0 - c * c))
    n = v * c + np.array([-rl2, rl2]) * h * v[[1, 0]]
    t1 = c1 + r * n
    t2 = c2 + rl1 * r * n
    return np.array([t1, t2])


def get_dubins(r, u, v):
    cuR, cuL, cvR, cvL = tuple(get_circs_from_points(r, u, v))
    directions = [(cuR, cvR, 0), (cuL, cvL, 1), (cuR, cvL, 2), (cuL, cvR, 3)]
    for (cu, cv, i) in directions:
        straight = get_tans(r, cu, cv, i)
        if straight is not None:
            tanp1 = straight[0, :]
            tanp2 = straight[1, :]
            S = np.diff(straight, axis=0).reshape(2)


def get_mpl_arrow(p, ax, color="k", width=0.04):
    p[0]
    p[1]
    dx = cos(p[2]) * 1
    dy = sin(p[2]) * 1
    w = width
    ax.arrow(
        p[0],
        p[1],
        dx,
        dy,
        color=color,
        width=w,
        head_width=10.0 * w,
        head_length=10.0 * w,
        fill=None,
    )


def make_dubins_plot(u, v, cu, cv, i, tanp1, S):
    axi = {0: (n, 0), 1: (n, 1), 2: (n + 1, 0), 3: (n + 1, 1)}
    lri = {0: "Right, Right", 1: "Left, Left", 2: "Right, Left", 3: "Left, Right"}
    ax[axi[i]].scatter(u[0], u[1], s=10, c="blue")
    ax[axi[i]].scatter(v[0], v[1], s=10, c="red")
    get_mpl_arrow(u, ax[axi[i]], color="blue", width=0.02)
    get_mpl_arrow(v, ax[axi[i]], color="red", width=0.02)
    ax[axi[i]].arrow(
        *tanp1,
        *S,
        width=0.02,
        head_width=10.0 * 0.02,
        head_length=10.0 * 0.02,
        fill=None
    )
    ax[axi[i]].add_artist(Circle(cu, r, fill=None, ec="b"))
    ax[axi[i]].add_artist(Circle(cv, r, fill=None, ec="r"))
    ax[axi[i]].set_xlim((min(u[0], v[0]) - 2 * r - 1.0, max(u[0], v[0]) + 2 * r + 1.0))
    ax[axi[i]].set_ylim((min(u[1], v[1]) - 2 * r - 1.0, max(u[1], v[1]) + 2 * r + 1.0))
    ax[axi[i]].set_aspect("equal")
    ax[axi[i]].get_yaxis().set_visible(False)
    ax[axi[i]].get_xaxis().set_visible(False)
    ax[axi[i]].set_title(lri[i])


if __name__ == "__main__":
    from matplotlib.patches import Circle
    import matplotlib.pyplot as plt

    for i in range(4):
        u = np.array([0.0, 0.0, 0.4])
        v = np.array([5.0, 5.0, -1.2])
        ptsu = np.random.uniform(0, 1, (1, 3)) * np.array([10.0, 10.0, P2I])
        ptsv = np.random.uniform(0, 1, (1, 3)) * np.array([10.0, 10.0, P2I])

        fig, ax = plt.subplots(nrows=ptsu.shape[0] * 2, ncols=2)
        for u, v, n in zip(ptsu, ptsv, range(0, ptsu.shape[0] * 2, 2)):
            r = 1.0
            get_dubins(r, u, v, ax, n)
        plt.show()
        # fig.savefig("images/straight_dubins_{}".format(i), dpi=300)
