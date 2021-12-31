import world_gen
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib import cm
from rrt import RRTstar, get_rrt_LC
import numba as nb
from math import sqrt
from scipy import interpolate


def make_vfield(world, sigma=5.0):
    gf = gaussian_filter(np.float64(world), sigma=sigma)
    diffx = np.diff(gf, n=1, axis=0, prepend=0)
    diffy = -np.diff(gf, n=1, axis=1, prepend=0)
    theta = np.arctan2(diffy, diffx)
    xy = np.stack((diffx, diffy), axis=-1)
    r = np.linalg.norm(xy, axis=-1)
    return np.stack((theta, r), axis=-1), xy


@nb.njit(fastmath=True)
def r2norm(p1, p2):
    v = p2 - p1
    return sqrt(v[0] * v[0] + v[1] * v[1])


def calc_directions(T):
    """get normal vector for each edge in T"""
    T.nodes[0]["direction"] = np.array((0.0, 0.0))
    for e1, e2 in T.edges:
        p1 = T.nodes[e1]["point"]
        p2 = T.nodes[e2]["point"]
        # direction is attached to p2.
        norm = r2norm(p1, p2)
        if norm <= 1e-2:
            direction = np.array((1.0, 0.0))
        else:
            direction = (p2 - p1) / norm
        T[e1][e2]["direction"] = direction
        T.nodes[e2]["direction"] = direction
    return T


def plot_directions(T, ax):
    # TODO refactor
    points = []
    directions = []
    for v in T.nodes:
        point = T.nodes[v]["point"]
        direction = T.nodes[v]["direction"]
        points.append(point)
        directions.append(direction)
    points = np.array(points)
    directions = np.array(directions)
    ax.quiver(points[:, 0], points[:, 1], directions[:, 1], directions[:, 0])
    return ax


def cost_of_nearest(vcosts, points, xy):
    dists = np.linalg.norm(points - xy, axis=-1)
    return vcosts[np.argmin(dists)]


def make_grad_field(vcosts, points, world_shape):
    scalar_field = np.full(world.shape, -10)
    for xy in np.argwhere(world == 0):
        cnearest = cost_of_nearest(vcosts, points, xy)
        scalar_field[xy[0], xy[1]] = cnearest
    return scalar_field


def get_cost_vfield(gf, sigma=5.0):
    gf = gaussian_filter(gf, sigma=sigma)
    diffx = np.diff(gf, n=1, axis=0, prepend=0)
    diffy = np.diff(gf, n=1, axis=1, prepend=0)
    theta = np.arctan2(diffy, diffx)
    xy = np.stack((diffx, diffy), axis=-1)
    r = np.linalg.norm(xy, axis=-1)
    return np.stack((theta, r), axis=-1), xy


def get_pos_arr(shape):
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    xy = np.meshgrid(x, y)
    return np.stack(xy, axis=-1)


if __name__ == "__main__":
    w, h = 512, 512
    world = world_gen.make_world((h, w), (4, 4))
    world = world | world_gen.make_world((h, w), (2, 2))
    xstart, xgoal = world_gen.get_rand_start_end(world)

    rrts = RRTstar(world, 1200)
    T = rrts.make(xstart, xgoal, float(w) / 4.0)
    T = calc_directions(T)

    gf = get_cost_vfield(make_grad_field(rrts.vcosts, rrts.points))

    xy = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1)
    vfield, vfieldxy = make_vfield(world)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.imshow(world.T, cmap=cm.get_cmap("Greys"), origin="lower")
    # ax.quiver(xy[:, :, 0].T, xy[:, :, 1].T, vfieldxy[:, :, 1], vfieldxy[:, :, 0])

    # ax.add_collection(get_rrt_LC(T))

    gf = make_grad_field(rrts.vcosts, rrts.points)
    ax.imshow(gf.T, cmap=cm.get_cmap("viridis"), origin="lower")
    gf, gfxy = get_cost_vfield(gf)

    ax.quiver(
        xy[:, :, 0].T,
        xy[:, :, 1].T,
        gfxy[:, :, 0] * 100,
        gfxy[:, :, 1] * 100,
        units="xy",
    )

    plt.show()
