from ctypes import ArgumentError
from scipy.spatial.distance import euclidean
import numpy as np
import networkx as nx
from scipy.linalg import svd, det, norm
from tqdm import tqdm
from matplotlib.patches import Ellipse


def dotself(u):
    """dot vector `u` with self"""
    return np.dot(u, u)


def make_pos(T: nx.DiGraph) -> dict:
    """extract a dict where keys are nodes and values are positions"""
    pos = {}
    for n in T.nodes:
        pos[n] = T.nodes[n]["point"]
    return pos


def get_rand_start_end(world, bias=True):
    """get random free start, end in the world"""
    free = np.argwhere(world == 0)
    """if bias, prefer points far away from one another"""
    if bias == True:
        start_i = int(np.random.beta(a=0.5, b=5) * free.shape[0])
        end_i = int(np.random.beta(a=5, b=0.5) * free.shape[0])
    else:
        start_i = np.random.choice(free.shape[0])
        end_i = np.random.choice(free.shape[0])
    start = free[start_i, :]
    end = free[end_i, :]
    return start, end


def nearest_nodes(T: nx.DiGraph, x) -> list:
    """get a list of the nodes sorted by distance to point `x`, nearest first"""

    def distance(u):
        x1 = T.nodes[u]["point"]
        x2 = x
        d2 = dotself(x1 - x2)
        if d2 != 0:
            return d2
        else:
            return np.Infinity

    return sorted([n for n in T.nodes], key=distance)


def get_end_node(T: nx.DiGraph, world, xgoal):
    """get the 'end' node, given a tree and an end point. The end node is either the point itself,
    if a path to it is possible, or the closest node in the tree to the end point."""
    vnearest = nearest_nodes(T, xgoal)[0]
    xnearest = T.nodes[vnearest]["point"]
    if not collision(T, world, xnearest, xgoal):
        v = max(T.nodes) + 1
        newcost = calc_cost(T, vnearest, xgoal)
        T.add_node(v, point=xgoal, cost=newcost)
        T.add_edge(vnearest, v, dist=euclidean(xnearest, xgoal), cost=newcost)
        return v
    else:
        return vnearest


def near(T: nx.DiGraph, x, r):
    """get nodes within `r` of point `x`"""
    r2 = r * r
    within = []
    for n in T.nodes:
        if dotself(T.nodes[n]["point"] - x) < r2:
            within.append(n)
    return within


def path(T: nx.DiGraph, startv, endv) -> list:
    """get path from one vertex to another, in the form of a list of nodes comprising the path"""
    path = nx.shortest_path(T, source=startv, target=endv, weight="dist")
    point_path = np.array([T.nodes[n]["point"] for n in path])
    return point_path


########### RRT Star Informed


def unitball():
    """draw a point from a uniform distribution bounded by the ball at 1 = x^2 + y^2"""
    r = np.random.uniform(0, 1)
    theta = 2 * np.pi * np.random.uniform(0, 1)
    x = np.sqrt(r) * np.cos(theta)
    y = np.sqrt(r) * np.sin(theta)
    unif = np.array([x, y])
    return unif


def calc_cost(T, v, x=None):
    if x is not None:
        return T.nodes[v]["cost"] + euclidean(T.nodes[v]["point"], x)
    else:
        return T.nodes[v]["cost"]


def get_parent(T, v):
    """get first parent of node `v` on T. If a parent does not exist,
    return None."""
    return next(T.predecessors(v), None)


def collision(T: nx.DiGraph, world, a, b) -> bool:
    x0 = a[0]
    y0 = a[1]
    x1 = b[0]
    y1 = b[1]

    dx = abs(x1 - x0)
    if x0 < x1:
        sx = 1
    else:
        sx = -1

    dy = -abs(y1 - y0)
    if y0 < y1:
        sy = 1
    else:
        sy = -1
    err = dx + dy
    while True:
        if world[x0, y0] == 1:
            return True
        elif x0 == x1 and y0 == y1:
            return False
        else:
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy


############ RRT STANDARD
def make_RRT_standard(world, xstart, xgoal, N):
    """Make RRT standard tree with `N` points from xstart to xgoal.
    Returns the tree, the start node, and the end node."""
    T = nx.DiGraph()
    i = 1
    vstart = i
    T.add_node(vstart, point=xstart)
    pbar = tqdm(total=N)
    while i < N:
        # uniform sample over world
        xnew = sample(world)
        vnearest = nearest_nodes(T, xnew)[0]
        xnearest = T.nodes[vnearest]["point"]
        if not collision(T, world, xnearest, xnew) and all(xnearest != xnew):
            T.add_node(i, point=xnew)
            T.add_edge(vnearest, i, dist=euclidean(xnearest, xnew))
            pbar.update(1)
            i += 1
    pbar.close()
    vend = get_end_node(T, world, xgoal)
    return T, vstart, vend


########## RRT Star
def make_RRT_star(world, xstart, xgoal, N, n_rewire):
    """Make RRT star with `N` points from xstart to xgoal.
    Returns the tree, the start node, and the end node."""
    T = nx.DiGraph()
    i = 1
    vstart = 0
    T.add_node(vstart, point=xstart, cost=0.0)
    pbar = tqdm(total=N)
    while i < N:
        xnew = sample(world)
        vnearest = nearest_nodes(T, xnew)
        xnearest = T.nodes[vnearest[0]]["point"]
        if not collision(T, world, xnearest, xnew):
            # add the node for sure
            vnew = i
            # now we look at edges...
            vmin = vnearest[0]
            xmin = xnearest
            cmin = calc_cost(T, vmin, xnew)
            # search other nodes for a lower cost than minimum
            for j, vn in enumerate(vnearest):
                coll = collision(T, world, T.nodes[vn]["point"], xnew)
                cost = calc_cost(T, vn, xnew)
                # lower cost and no collision
                if (cost < cmin) and (not coll):
                    xmin = T.nodes[vn]["point"]
                    cmin = cost
                    vmin = vn
                # constant-time search
                if j > n_rewire:
                    break
            dist = euclidean(T.nodes[vmin]["point"], xnew)
            T.add_node(vnew, point=xnew, cost=cmin)
            T.add_edge(vmin, vnew, dist=dist)
            # now rewire the tree
            for j, vn in enumerate(vnearest):
                coll = collision(T, world, T.nodes[vn]["point"], xnew)
                cost = calc_cost(T, vn, xnew) < calc_cost(T, vn)
                if (not coll) and cost:
                    vparent = get_parent(T, vn)
                    if vparent is not None:
                        T.remove_edge(vparent, vn)
                        T.add_edge(vnew, vn)
                if j > n_rewire:
                    break
            pbar.update(1)
            i += 1
    pbar.close()
    vend = get_end_node(T, world, xgoal)
    return T, vstart, vend


def make_RRT_starinformed(world, xstart, xgoal, N, r_rewire, rgoal):
    """make rrtstar informed"""
    Vsoln = set()
    T = nx.DiGraph()
    i = 1
    vstart = 0
    T.add_node(vstart, point=xstart, cost=0.0)
    pbar = tqdm(total=N)
    while i < N:
        vbest, cbest = least_cost(T, Vsoln)
        if vbest is not None:
            xrand = sample(
                world,
                (xstart, xgoal),
                cmax=cbest + euclidean(T.nodes[vbest]["point"], xgoal),
            )
        else:
            xrand = sample(world)
        vnearest = nearest_nodes(T, xrand)[0]
        xnew = xrand
        if not collision(T, world, T.nodes[vnearest]["point"], xnew):
            vnear = near(T, xnew, r_rewire)
            vnear.append(vnearest)
            vbest, new_cost = min(
                [
                    (v, calc_cost(T, v, xnew))
                    for v in vnear
                    if not collision(T, world, T.nodes[v]["point"], xnew)
                ],
                key=lambda t: t[1],
            )

            pbar.update(1)
            i += 1
            T.add_node(i, point=xnew, cost=new_cost)
            T.add_edge(vbest, i, cost=new_cost)

            for vn in vnear:
                xn = T.nodes[vn]["point"]
                if calc_cost(T, i, xn) < calc_cost(T, vn):
                    if not collision(T, world, xnew, xn):
                        vparent = get_parent(T, vn)
                        if vparent is not None:
                            T.remove_edge(vparent, vn)
                            rewired_cost = calc_cost(T, i, xn)
                            T.nodes[vn]["cost"] = rewired_cost
                            T.add_edge(i, vn, cost=rewired_cost)

            if dotself(xnew - xgoal) < rgoal * rgoal:
                Vsoln.add(i)

    pbar.close()
    if len(Vsoln) > 0:
        vend, cbest = min([(v, T.nodes[v]["cost"]) for v in Vsoln], key=lambda t: t[1])
        ell = get_ellipse_ax(
            xstart, xgoal, cbest, euclidean(T.nodes[vend]["point"], xgoal)
        )
    else:
        ell = Ellipse((0, 0), 1, 1, 0)
        vend = nearest_nodes(T, xgoal)[0]

    return T, vstart, vend, ell


def rad2deg(a):
    return a * 180 / np.pi


def rotation_to_world_frame(xstart, xgoal):
    """calculate the rotation matrix from the world-frame to the frame given
    by the hyperellipsoid with focal points at xf1=xstart and xf2=xgoal. a unit
    ball multiplied by this matrix will produce an oriented ellipsoid with those
    focal points."""
    a1 = np.atleast_2d((xgoal - xstart) / norm(xgoal - xstart))
    M = np.outer(a1, np.atleast_2d([1, 0]))
    U, _, V = svd(M)
    return U @ np.diag([det(U), det(V)]) @ V.T


def get_ellipse_xform(xstart, xgoal, cmax, d):
    """transform vector in unit plane to ellipse plane"""
    C = rotation_to_world_frame(xstart, xgoal)
    r1 = cmax / 2
    r2 = np.sqrt(abs(dotself(cmax + d) - dotself(xstart - xgoal))) / 2
    print(r2)
    L = np.diag([r1, r2])
    return np.dot(C, L)


def get_ellipse_ax(xstart, xgoal, cbest, d):
    xcent = (xgoal + xstart) / 2

    # get the rotation
    CL = get_ellipse_xform(xstart, xgoal, cbest, d)

    # apply the rotation to find major axis, minor axis, angle
    a = np.dot(CL, np.array([1, 0]))
    b = np.dot(CL, np.array([0, 1]))
    majax = 2 * norm(a)
    minax = 2 * norm(b)
    ang = rad2deg(np.arctan2((a)[1], (a)[0]))

    return Ellipse(xcent, majax, minax, ang, fill=None)


def sample(world, startgoal=None, cmax=None, vmax=0):
    """sample from free space in the world. If cmax and startgoal are given, sample from
    the ellipsoid with focal points xstart and xgoal"""
    free = np.argwhere(world == 0)
    if cmax is not None and startgoal is not None:
        xstart, xgoal = startgoal
        xcent = (xstart + xgoal) / 2
        CL = get_ellipse_xform(xstart, xgoal, cmax, vmax)
        xball = unitball()

        x, y = tuple(np.dot(CL, xball) + xcent)
        # clamp to finite world
        x = int(max(0, min(world.shape[0] - 1, x)))
        y = int(max(0, min(world.shape[1] - 1, y)))
        point = np.array((x, y))
        return point
    else:
        return free[np.random.choice(free.shape[0])]


def least_cost(T, V):
    """get lowest cost from a collection `V` of vertices in T"""
    if len(V) == 0:
        return None, None
    else:
        return min([(v, T.nodes[v]["cost"]) for v in V], key=lambda t: t[1])
