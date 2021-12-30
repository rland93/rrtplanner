import numpy as np
import numba as nb
from rrt import RRTstar
import networkx as nx
import pyvoronoi
from collections import defaultdict
from shapely import geometry
from shapely.ops import unary_union
from tqdm import tqdm
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from rrtpp.rrt import get_rrt_LC


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


def get_voronoi_regions(path) -> dict:
    pv = pyvoronoi.Pyvoronoi(len(path) * 5)
    segmask, segs = [], []

    for p1, p2 in path:
        pv.AddSegment([p1, p2])
        # we will actually use this segment
        segmask.append(True)
        segs.append([p1, p2])

        # now, we reflect across x=0, y=0, x=w, y=h
        # so that voronoi cells are finite
        trans = [0, w, 0, h]
        rot1 = [0, 0, 1, 1]
        rot2 = [1, 1, 0, 0]
        for t, r1, r2 in zip(trans, rot1, rot2):
            a, b = [None, None], [None, None]
            a[r1] = p1[r1] + ((t - p1) * 2)[r1]
            b[r1] = p2[r1] + ((t - p2) * 2)[r1]
            a[r2] = p1[r2]
            b[r2] = p2[r2]
            pv.AddSegment([a, b])
            # we will discard these (hence segmask == False)
            # after the voronoi cells are made, but we still add
            # them so that we can use the voronoi indices with `segs` array
            segmask.append(False)
            segs.append([a, b])

    segmask = np.array(segmask)

    # create the voronoi and unpack
    pv.Construct()
    cells = pv.GetCells()
    edges = pv.GetEdges()
    vertices = pv.GetVertices()

    # we unpack the voronoi cells
    rawpolys = defaultdict(list)
    for i, c in tqdm(enumerate(cells), desc="create voronoi cells"):
        cell_lines = []
        # only use non-reflected cells
        if segmask[c.site]:
            for edge in c.edges:
                # get the two points of the edge
                p1 = vertices[edges[edge].start]
                p2 = vertices[edges[edge].end]
                # unpack to np arr
                p1 = np.array((p1.X, p1.Y))
                p2 = np.array((p2.X, p2.Y))
                # now cell lines are made from those points
                cell_lines.append((p1, p2))
            # list -> np arr
            cell_lines = np.array(cell_lines)
            # cell lines are (2,2) arras, but we actually want
            # a list of points to construct shapely geometry. So
            # array transformed from (M x 2 x 2) -> (M * 2 x 2)
            # with x, y on axis=-1.
            cell_points = cell_lines.reshape(cell_lines.shape[0] * 2, 2)
            # each line is joined with the next line in the polygon.
            # so let's say we have a poly with vertices [0,0], [1,0], [1,1]
            # that is actually a collection of 3 2x2 arrays, one for each line:
            # [ [0,0],[1,0] ], [ [1,0], [1,1] ], [ [1,1], [0,0]]. When unpacked,
            # this is flattened to just the points:
            # [ [0,0], [1,0], [1,0], [1,1], [1,1], [0,0] ] this array of points
            # has duplicates for every other element, so we remove those duplicates.
            cell_points = cell_points[::2, :]
            # make a polygon from the points
            rawpolys[c.site].append(geometry.Polygon(cell_points))

    # store polygons into a nicer format
    polys = {}

    # integer points
    points = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1)
    geom_points = geometry.MultiPoint(points.reshape(w * h, 2))

    for k, v in tqdm(rawpolys.items(), desc="unpack voronoi cells"):
        # get hull
        polygons = geometry.MultiPolygon(v)
        union = unary_union(polygons)

        # get integer points that are in polygon
        points_in_poly = polygons.intersection(geom_points)
        points_in_poly = np.squeeze(
            np.array([np.int64(p.xy) for p in points_in_poly.geoms])
        )
        # get dists to line for each xy
        d2l = dist2line(segs[k][0], segs[k][1])
        dists = np.apply_along_axis(d2l, 1, points_in_poly)

        # get chi for this segment
        seg = np.array(segs[k])
        chi = np.arctan2(seg[1, 1] - seg[0, 1], seg[1, 0] - seg[0, 0])

        # get lines of hull
        union_lines = np.array(union.exterior.coords)

        polys[k] = {
            "cell": union,
            "cell_lines": union_lines,
            "seg": seg,
            "xys": points_in_poly,
            "dists": dists,
            "chi": chi,
        }
    return polys


def get_angles(polys, k=0.2):
    dists = np.zeros((w, h))
    angles = np.zeros((w, h))
    points = integer_points((w, h))

    for poly in tqdm(polys.values(), desc="calculate angles"):
        course = -np.arctan(k * poly["dists"]) + poly["chi"]
        angles[poly["xys"][:, 0], poly["xys"][:, 1]] = course
        dists[poly["xys"][:, 0], poly["xys"][:, 1]] = poly["dists"]

    return dists, angles, points


def integer_points(wh):
    return np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1)


def plot_angles(
    ax,
    dists,
    angles,
    points,
    cell_lines,
    vmin=-100,
    vmax=100,
):
    norm = Normalize(vmin=vmin, vmax=vmax)
    ### Distances

    ax.imshow(dists.T, cmap="RdBu", norm=norm)
    ### Arrows
    ax.quiver(
        points[:, :, 0],
        points[:, :, 1],
        np.cos(angles.T),
        np.sin(angles.T),
        scale=1.0,
        width=0.1,
        units="xy",
    )
    ### Cells
    ax.add_collection(LineCollection(cell_lines, color="olivedrab", alpha=0.5))


def plot_path(ax, segment_lines, color="r"):
    ax.add_collection(LineCollection(segment_lines, color=color))


def plot_RRT(ax, xstart, xgoal, wh, rrt_T=None, world=None):
    ax.scatter(xstart[0], xstart[1], c="r", s=50, label="start", marker="o")
    ax.scatter(xgoal[0], xgoal[1], c="b", s=50, label="goal", marker="*")
    if world is not None:
        ax.imshow(world.T, cmap="Greys", origin="lower")
    if rrt_T is not None:
        lc = get_rrt_LC(T)
        ax.add_collection(lc)
    ax.set_xlim(0, wh[0])
    ax.set_ylim(0, wh[1])


if __name__ == "__main__":
    from rrtpp.world_gen import make_world, get_rand_start_end
    from rrtpp import RRTstar
    from matplotlib import pyplot as plt

    fig1 = plt.figure()
    fig2 = plt.figure()

    # make world
    w, h = 64, 64
    world = make_world((w, h), (int(w / 32), int(h / 32)))
    world = world | make_world((w, h), (int(4), int(4)))
    xstart, xgoal = get_rand_start_end(world)

    # make RRTS
    rrts = RRTstar(world, 600)
    T, gv = rrts.make(xstart, xgoal, 256)

    # get voronoi regions
    path = nx.shortest_path(T, 0, gv, weight="dist")
    edges = []
    for i in range(len(path) - 1):
        p1 = T.nodes[path[i]]["pos"]
        p2 = T.nodes[path[i + 1]]["pos"]
        edges.append((p1, p2))

    polys = get_voronoi_regions(edges)
    dists, angles, points = get_angles(polys)

    ax = fig1.add_subplot(), fig2.add_subplot()

    plot_RRT(ax[0], xstart, xgoal, world.shape, T, world)
    plot_RRT(ax[1], xstart, xgoal, world.shape)

    plot_path(ax[1], edges)
    plot_path(ax[0], edges)

    plot_angles(ax[1], dists, angles, points, [v["cell_lines"] for v in polys.values()])

    plt.show()