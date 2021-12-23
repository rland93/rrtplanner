import matplotlib.pyplot as plt
from math import cos, sin, sqrt, atan2, acos
import numpy as np
from tqdm import tqdm
from rrt import sample_all_free
import numba
from enum import Enum
from matplotlib.patches import Circle
from matplotlib.pyplot import arrow
import networkx as nx
import random
import time
from scipy.ndimage import binary_dilation


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
DIR_IDX = {
    0: (-1.0, -1.0),  # right, right
    1: (1.0, 1.0),  # left, left
    2: (-1.0, 1.0),  # right, left
    3: (1.0, -1.0),  # left, right
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


@numba.njit(fastmath=True)
def ang_2_vecs(a, b):
    return atan2(b[1], b[0]) - atan2(a[1], a[0])


def calc_angle(direction, p1, p2, c):
    v1 = p1 - c
    v2 = p2 - c
    diff = (atan2(v2[1], v2[0]) - atan2(v1[1], v1[0])) % P2I
    if direction > 0:
        return diff
    else:
        return -(P2I - diff)


def get_dubins_paths(r, u, v):
    cuR, cuL, cvR, cvL = tuple(get_circs_from_points(r, u, v))
    directions = [(cuR, cvR, 0), (cuL, cvL, 1), (cuR, cvL, 2), (cuL, cvR, 3)]
    for (cu, cv, i) in directions:
        straight = get_tans(r, cu, cv, i)
        if straight is not None:
            tanp1 = straight[0, :]
            tanp2 = straight[1, :]
            S = np.diff(straight, axis=0).reshape(2)
            # theta1 = turn angle 1
            # theta2 = turn angle 2
            theta1 = calc_angle(DIR_IDX[i][0], u[:2], tanp1, cu)
            theta2 = calc_angle(DIR_IDX[i][1], tanp2, v[:2], cv)
            # arc1 length, straight segment length, arc2 length
            arc1_l = abs(theta1 * r)
            stra_l = abs(r2norm(S))
            arc3_l = abs(theta2 * r)
            # total path length
            lengths = arc1_l, stra_l, arc3_l
            circs = (cu, cv, None)
            tanpts = (tanp1, tanp2)
            angles = (theta1, theta2, None)
            dirs = i, None
            yield lengths, circs, tanpts, angles, dirs
    for (cu, cv, i) in directions:
        if i == 0 or i == 1:
            cucvd = r2norm(cu - cv)
            if cucvd > 2 * r and cucvd < 4 * r:
                psi = acos(cucvd / (4 * r))
                psi_u = atan2((cv - cu)[1], (cv - cu)[0])
                for ccdir in (1.0, -1.0):
                    theta = psi_u + ccdir * psi
                    c3 = cu[:2] + 2 * r * np.array([cos(theta), sin(theta)])
                    # tangent points
                    tanp1 = (c3 - cu) / 2 + cu
                    tanp2 = (c3 - cv) / 2 + cv
                    # arc lengths
                    theta1 = calc_angle(DIR_IDX[i][0], u[:2], tanp1, cu)
                    theta2 = calc_angle(DIR_IDX[i][1], tanp2, v[:2], cv)
                    theta3 = calc_angle(-DIR_IDX[i][0], tanp1, tanp2, c3)

                    arc1_l = abs(theta1 * r)
                    arc2_l = abs(r * theta3)
                    arc3_l = abs(theta2 * r)
                    # total path length
                    lengths = arc1_l, arc2_l, arc3_l
                    circs = (cu, cv, c3)
                    tanpts = (tanp1, tanp2)
                    angles = (theta1, theta2, theta3)
                    dirs = i, ccdir
                    yield lengths, circs, tanpts, angles, dirs


def discretize_dubins_path(lengths, circs, tanpts, angles, dirs, r, u, v, n=40):
    """yield a discrete set of points for a dubins curve."""
    pts = []
    # first segment
    s0 = u[:2] - circs[0]
    sweep = atan2(s0[1], s0[0])
    for _ in range(n):
        pt = circs[0] + r * np.array([cos(sweep), sin(sweep)])
        pts.append(pt)
        sweep += angles[0] / n

    # midsection
    if circs[2] is not None:
        s2 = tanpts[0] - circs[2]
        sweep = atan2(s2[1], s2[0])
        for _ in range(n):
            pt = circs[2] + r * np.array([cos(sweep), sin(sweep)])
            pts.append(pt)
            sweep += angles[2] / n

    # second segment
    s1 = tanpts[1] - circs[1]
    sweep = atan2(s1[1], s1[0])
    for _ in range(n):
        pt = circs[1] + r * np.array([cos(sweep), sin(sweep)])
        pts.append(pt)
        sweep += angles[1] / n
    return np.array(pts[::-1])


def dubins_collision_free(dpaths, r, world):
    viable = []
    for dpath in dpaths:
        if not dubins_path_check(*dpath, r, world):
            viable.append(dpath)
    return sorted(viable, key=lambda dpath: sum(dpath[0]))


def dubins_path_check(lengths, circs, tanpts, angles, dirs, r, world):
    if circ_coll(*circs[0], r, world):
        return True
    if circ_coll(*circs[1], r, world):
        return True
    if circs[2] is None:
        tp1, tp2 = tanpts
        tp1x = clamp(int(tp1[0]), world.shape[0] - 1)
        tp1y = clamp(int(tp1[1]), world.shape[1] - 1)
        tp2x = clamp(int(tp2[0]), world.shape[0] - 1)
        tp2y = clamp(int(tp2[1]), world.shape[1] - 1)
        if line_coll(tp1x, tp1y, tp2x, tp2y, world):
            return True
    else:
        if circ_coll(*circs[2], r, world):
            return True
    return False


@numba.njit()
def clamp(m, shape):
    return min(max(m, 0), shape)


@numba.njit()
def mirror8(x, y):
    pts = []
    for i in (-1, 1):
        for j in (-1, 1):
            pts.append((i * x, j * y))
            pts.append((i * y, j * x))
    return pts


@numba.njit()
def bres_circ(r):
    points = []
    x = np.int64(0)
    y = -r
    err = 1 - r
    de = 3
    dne = -(r << 1) + 5
    points.extend(mirror8(x, y))
    while x < -y:
        if err <= 0:
            err += de
        else:
            err += dne
            dne += 2
            y += 1
        de += 2
        dne += 2
        x += 1
        points.extend(mirror8(x, y))
    return points


@numba.njit()
def line_coll(x1, y1, x2, y2, world):
    dx = abs(x2 - x1)
    if x1 < x2:
        sx = 1
    else:
        sx = -1
    dy = -abs(y2 - y1)
    if y1 < y2:
        sy = 1
    else:
        sy = -1
    err = dx + dy
    while True:
        if world[x1, y1] == 1:
            return True
        elif x1 == x2 and y1 == y2:
            return False
        else:
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x1 += sx
            if e2 <= dx:
                err += dx
                y1 += sy


def circ_coll(cx, cy, r, world):
    circ = bres_circ(np.int64(r))
    circ_mask = np.zeros_like(world, dtype=bool)
    for (x, y) in circ:
        x_ = clamp(int(cx) + x, world.shape[0] - 1)
        y_ = clamp(int(cy) + y, world.shape[1] - 1)
        circ_mask[x_, y_] = 1
    if np.any(world[circ_mask]):
        return True
    else:
        return False


def plot_dubins(ax, r, u, v, lengths, circs, tanpts, angles, dirs, buf=3.0):
    c1, c2, c3 = circs
    tanp1, tanp2 = tanpts
    # plot points
    ax.scatter(u[0], u[1], s=10, c="blue")
    ax.scatter(v[0], v[1], s=10, c="red")
    get_mpl_arrow(u, ax, color="blue", width=0.02)
    get_mpl_arrow(v, ax, color="red", width=0.02)
    # draw circles
    ax.add_artist(Circle(c1, r, fill=None, ec="silver"))
    ax.add_artist(Circle(c2, r, fill=None, ec="silver"))
    # RLR, LRL case
    if c3 is not None:
        # no straight path only circles
        ax.add_artist(Circle(c3, r, fill=None, ec="silver"))
        ax.scatter(tanp1[0], tanp1[1], s=10, c="purple")
        ax.scatter(tanp2[0], tanp2[1], s=10, c="orange")
    else:
        # draw straight paths
        # ax.add_artist(Line2D((tanp1[0], tanp2[0]), (tanp1[1], tanp2[1])))
        ax.scatter(tanp1[0], tanp1[1], s=10, c="purple")
        ax.scatter(tanp2[0], tanp2[1], s=10, c="orange")

    ax.set_xlim((min(u[0], v[0]) - 2 * r - buf, max(u[0], v[0]) + 2 * r + buf))
    ax.set_ylim((min(u[1], v[1]) - 2 * r - buf, max(u[1], v[1]) + 2 * r + buf))
    ax.set_aspect("equal")
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)


def get_mpl_arrow(p, ax, color="k", size=1.0):
    p[0]
    p[1]
    dx = cos(p[2]) * size
    dy = sin(p[2]) * size
    ax.arrow(
        p[0],
        p[1],
        dx,
        dy,
        color=color,
        width=size * 0.05,
        head_width=10.0 * size * 0.1,
        head_length=10.0 * size * 0.2,
        fill=None,
    )


def nearest(points, x):
    near = []
    for p in points:
        near.append(r2norm(p - x))
    return near.index(min(near))


def make_circ_kernel(r):
    kern = np.zeros((int(r) * 2, int(r) * 2))
    center = int(r), int(r)
    for (x, y) in np.ndindex(kern.shape):
        dx = x - center[0]
        dy = y - center[1]
        if dx * dx < r * r and dy * dy < r * r:
            kern[x, y] = 1
    return kern


class RRTDubins(object):
    def __init__(self, world, radius):
        self.points = []
        self.edges = []
        self.headings = []
        self.twists = []
        self.radius = radius
        self.vcosts = []
        self.world = world
        self.worlddi = binary_dilation(world, make_circ_kernel(radius))
        self.every = 10
        self.free = np.argwhere(self.worlddi == 0)

    @staticmethod
    def get_twist(x, h):
        twist = np.empty((3,))
        twist[:2] = x
        twist[2] = h
        return twist

    def try2reachgoal(self, ptwist, gtwist):
        dpaths = get_dubins_paths(self.radius, ptwist, gtwist)
        viable = dubins_collision_free(dpaths, self.radius, self.world)
        if len(viable) > 0:
            lcpath = min(viable, key=lambda path: sum(path[0]))
            return lcpath
        else:
            return None

    def make(self, xstart, xgoal, hstart):
        # store xstart into points
        self.points.append(xstart)
        self.headings.append(hstart)
        tstart = self.get_twist(xstart, hstart)
        self.twists.append(tstart)
        # dpath from i -> j is stored in key (i, j)
        dpaths = {}

        reached_goal = False
        i = 0
        T = nx.DiGraph()
        T.add_node(i, points=xstart, heading=hstart, twist=tstart)
        t1 = time.time()
        while not reached_goal:
            xnew = sample_all_free(self.free)
            hnew = random.uniform(0, P2I)
            tnew = self.get_twist(xnew, hnew)
            vnearest = nearest(self.points, xnew)
            try_paths = get_dubins_paths(self.radius, self.twists[vnearest], tnew)
            viable_paths = dubins_collision_free(try_paths, self.radius, self.world)
            if len(viable_paths) > 0:
                # add this path
                vnew = len(self.points)
                self.points.append(xnew)
                cost = self.points[vnearest] + r2norm(xnew - self.points[vnearest])
                T.add_node(vnew, point=xnew, heading=hnew, twist=tnew, cost=cost)
                self.twists.append(self.get_twist(xnew, hnew))
                self.edges.append((vnearest, vnew))

                # least cost path
                lcpath = min(viable_paths, key=lambda path: sum(path[0]))
                dpath_disc = discretize_dubins_path(
                    *lcpath, self.radius, self.twists[vnearest], tnew
                )
                dpaths[(vnearest, vnew)] = {"discretized": dpath_disc, "path": lcpath}

                if i % self.every:
                    gtwist = self.get_twist(xgoal, random.uniform(0, P2I))
                    goalpath = self.try2reachgoal(tnew, gtwist)
                    if goalpath is not None:
                        vgoal = len(self.points)
                        gv = vgoal
                        self.points.append(xgoal)
                        dpath_disc = discretize_dubins_path(
                            *goalpath, self.radius, tnew, gtwist
                        )
                        dpaths[(vnew, vgoal)] = {
                            "discretized": dpath_disc,
                            "path": lcpath,
                        }
                        self.twists.append(gtwist)
                        # reached_goal = True
            if i > 400:
                # print("RRT failed to reach goal")
                gv = nearest(self.points, xgoal)
                break
            i += 1

        for (i, j), dpath in dpaths.items():
            T.add_edge(i, j, **dpath)

        print("Took {}s.".format(time.time() - t1))

        return T, gv


def near(points, x, r):
    """get idx of near points"""
    near = []
    for i, p in enumerate(points):
        if r2norm(p - x) < r:
            near.append(i)
    return near


class RRTStarDubins(RRTDubins):
    def __init__(self, world, radius, r_rewire):
        super().__init__(world, radius)
        self.r_rewire = r_rewire
        self.every = 100
        self.pbar = True
        self.tries = 1000

    def near(self, x, r):
        """get nodes within `r` of point `x`"""

        dists = np.linalg.norm(self.points - x, axis=1)
        return np.argwhere(dists < r)

    def get_cost(self, v, x):
        return self.vcosts[v] + r2norm(self.points[v] - x)

    def get_parent(self, v):
        for i, (e1, e2) in enumerate(self.edges):
            if e2 == v:
                return i
        return None

    def make(self, xstart, xgoal, hstart):
        self.points.append(xstart)
        self.headings.append(hstart)
        tstart = self.get_twist(xstart, hstart)
        self.twists.append(tstart)
        self.vcosts.append(0.0)

        dpaths = {}
        reached_goal = False
        i = 0
        T = nx.DiGraph()
        T.add_node(i, points=xstart, heading=hstart, twist=tstart)

        if self.pbar:
            pbar = tqdm(total=self.tries + 3, desc="tree")
        while not reached_goal:
            xnew = sample_all_free(self.free)
            hnew = random.uniform(0, P2I)
            tnew = self.get_twist(xnew, hnew)
            vnearest = nearest(self.points, xnew)
            try_paths = get_dubins_paths(self.radius, self.twists[vnearest], tnew)
            viable_paths = dubins_collision_free(try_paths, self.radius, self.world)
            if len(viable_paths) > 0:
                vnew = len(self.points)
                # get near points
                vnears = near(self.points, xnew, self.r_rewire)

                # calculate best cost
                cbest = self.get_cost(vnearest, xnew)
                vbest = vnearest
                for vnear in vnears:
                    xnear = self.points[vnear]
                    cnear = self.get_cost(vnear, xnew)
                    if cnear < cbest:
                        # get dubins collision
                        try_paths = get_dubins_paths(
                            self.radius, self.twists[vnear], tnew
                        )
                        viable_paths = dubins_collision_free(
                            try_paths, self.radius, self.world
                        )
                        if len(viable_paths) > 0:
                            cbest = cnear
                            vbest = vnear

                # store new point, edge, dubins path, etc.
                try_paths = get_dubins_paths(self.radius, self.twists[vbest], tnew)
                viable_paths = dubins_collision_free(try_paths, self.radius, self.world)
                lcpath = min(viable_paths, key=lambda path: sum(path[0]))

                self.points.append(xnew)
                self.twists.append(tnew)
                self.headings.append(hnew)
                self.edges.append((vbest, vnew))
                self.vcosts.append(cbest)

                dpath_disc = discretize_dubins_path(
                    *lcpath, self.radius, self.twists[vbest], tnew
                )
                dpaths[(vbest, vnew)] = {"discretized": dpath_disc, "path": lcpath}

                # rewire the tree now
                for vnear in vnears:
                    xnear = self.points[vnear]
                    cnear = self.vcosts[vnear]
                    possible_cost = self.get_cost(vnear, xnew)
                    if possible_cost < cnear:
                        try_paths = get_dubins_paths(
                            self.radius, self.twists[vnear], tnew
                        )
                        viable_paths = dubins_collision_free(
                            try_paths, self.radius, self.world
                        )
                        if len(viable_paths) > 0:
                            # get parent of vnear
                            parent = self.get_parent(vnear)
                            if parent is not None:
                                old_edge = self.edges[parent]
                                self.edges[parent] = (vnew, vnear)
                                self.vcosts[vnear] = possible_cost
                                lcpath = min(
                                    viable_paths, key=lambda path: sum(path[0])
                                )
                                dpath_disc = discretize_dubins_path(
                                    *lcpath, self.radius, self.twists[vnear], tnew
                                )
                                # rewrite dpaths
                                dpaths[old_edge] = {
                                    "discretized": dpath_disc,
                                    "path": lcpath,
                                }
                # attempt to go to goal
                if i % self.every == 0:
                    gtwist = self.get_twist(xgoal, random.uniform(0, P2I))
                    goalpath = self.try2reachgoal(tnew, gtwist)
                    if goalpath is not None:
                        vgoal = len(self.points)
                        gv = vgoal
                        self.points.append(xgoal)
                        dpath_disc = discretize_dubins_path(
                            *goalpath, self.radius, tnew, gtwist
                        )
                        dpaths[(vnew, vgoal)] = {
                            "discretized": dpath_disc,
                            "path": lcpath,
                        }
                        self.twists.append(gtwist)
                        self.edges.append((vnew, gv))
                        self.vcosts.append(self.get_cost(vnew, xgoal))

                        reached_goal = True
            if self.pbar:
                pbar.update(1)
            if i > self.tries:
                # print("RRT failed to reach goal")
                gv = nearest(self.points, xgoal)
                break
            i += 1
        if self.pbar:
            pbar.update(1)
            pbar.close()
        for (i, j), dpath in dpaths.items():
            T.add_edge(i, j, **dpath)
        return T, gv


def gotoroot(T: nx.DiGraph, leaf, path=[]):
    """go to root of the tree from a leaf"""
    try:
        parent = T.predecessors(leaf).__next__()
    except StopIteration:
        return path
    path.append((parent, leaf))
    return gotoroot(T, parent, path)


def get_directive(T: nx.DiGraph, edges):
    """from a series of vertex edges, get directive"""
    for e1, e2 in edges:
        dpath = T[e1][e2]["path"]
        lengths, circs, tanpts, angles, dirs = dpath
        print(angles)


if __name__ == "__main__":
    from world_gen import make_world, get_rand_start_end
    from matplotlib import cm
    from matplotlib.collections import LineCollection

    wsize = (1024, 1024)
    # world = np.zeros(wsize, dtype=bool)
    world = make_world(wsize, (2, 2))
    world |= make_world(wsize, (4, 4))

    xstart, xgoal = get_rand_start_end(world)
    hstart = PI

    rrtd = RRTStarDubins(world, 7.0, 256.0)
    T, gv = rrtd.make(xstart, xgoal, hstart)

    # rrtd = RRTDubins(world, 16.0)
    # T, gv = rrtd.make(xstart, xgoal, hstart, 64)

    path = gotoroot(T, gv)
    path = list(reversed(path))

    fig, ax = plt.subplots(dpi=150, figsize=(7, 3))

    ax.imshow(world.T, cmap=cm.get_cmap("Greys"))

    dpaths = [T[e1][e2]["discretized"] for (e1, e2) in T.edges()]
    lc = LineCollection(dpaths, color="tan")
    lc.set_label("Path")
    ax.add_collection(lc)

    twists = np.array(rrtd.twists)
    points = np.array(rrtd.points)

    ax.quiver(
        points[:, 0],
        points[:, 1],
        np.cos(twists[:, 2]),
        np.sin(twists[:, 2]),
        color="lightsteelblue",
        label="Sample Points",
        angles="xy",
        scale_units="dots",
        scale=0.05,
    )
    ax.scatter(
        xstart[0], xstart[1], s=50, marker=".", c="orangered", label="Start", zorder=10
    )
    ax.scatter(
        xgoal[0], xgoal[1], s=50, marker="*", c="dodgerblue", label="Goal", zorder=10
    )

    path = gotoroot(T, gv)
    pathline = []
    for (i, j) in reversed(path):
        pathline.append(T[i][j]["discretized"])
    pathlc = LineCollection(pathline, color="blue")
    ax.add_collection(pathlc)
    pathlc.set_label("Path")

    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")

    plt.show()
