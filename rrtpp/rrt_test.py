from . import rrt
import networkx as nx
import numpy as np
from perlin_numpy import generate_perlin_noise_2d


def make_spatial_graph_like(n):
    T = nx.DiGraph()
    toadd = ((r, {"point": np.random.uniform(0, 100, size=(2,))}) for r in range(n))
    T.add_nodes_from(toadd)
    return T


def make_world(shape, empty=True, dtype=bool):
    if empty:
        # just return all 0's
        return np.zeros(shape=shape, dtype=dtype)
    else:
        # generate noise
        w, h = shape
        noise1 = generate_perlin_noise_2d((w, h), (4, 4), tileable=(True, True))
        noise2 = generate_perlin_noise_2d((w, h), (2, 2), tileable=(True, True))
        noise = (noise1 + noise2) / 2
        noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
        threshed = np.where(noise < 0.33, 1, 0)
        # cast to dtype
        typed_arr = np.empty_like(threshed, dtype=dtype)
        typed_arr = threshed
        return typed_arr


def make_world_types():
    emptys = [True, False]
    shapes = [(32, 32), (64, 64), (256, 256)]
    dtypes = [bool, int, np.uint8, np.uint32]
    for e in emptys:
        for s in shapes:
            for d in dtypes:
                yield make_world(s, empty=e, dtype=d)


def test_dotself():
    u = 1
    assert rrt.dotself(u) == 1
    v = [1, 0]
    assert rrt.dotself(v) == 1
    w = [0, 1]
    assert rrt.dotself(w) == 1
    x = [1, 1]
    assert rrt.dotself(x) == 2


def test_make_pos():
    n = 100
    T = make_spatial_graph_like(n)
    pos = rrt.make_pos(T)
    assert len(pos) == n


def test_get_rand_start_end():
    for b in [True, False]:
        for w in make_world_types():
            xstart, xend = rrt.get_rand_start_end(w, bias=b)
            assert 0 <= xstart[0] and xstart[0] <= w.shape[0]
            assert 0 <= xstart[1] and xstart[1] <= w.shape[1]


def test_nearest_nodes():
    n = 100
    xs = np.random.uniform(0, 1e3, size=(n, 2))
    T = make_spatial_graph_like(n)
    for x in xs:
        Vnearest = rrt.nearest_nodes(T, x)
        assert len(Vnearest) == len(T)
        d1 = rrt.dotself(T.nodes[Vnearest[0]]["point"] - x)
        d2 = rrt.dotself(T.nodes[Vnearest[-1]]["point"] - x)
        assert d1 < d2


def test_each_rrt():
    ns = [10, 100, 400]
    r_rewires = [5, 10, 20]
    r_goals = [2, 5, 10]
    for w in make_world_types():
        xstart, xgoal = rrt.get_rand_start_end(w)
        for (n, r_rewire, r_goal) in zip(ns, r_rewires, r_goals):
            T1, v1start, v1end = rrt.make_RRT_standard(w, xstart, xgoal, n)
            # T2, v2start, v2end = rrt.make_RRT_star(w, xstart, xgoal, n, r_rewire)
            # T3, v3start, v3end = rrt.make_RRT_starinformed(
            #     w, xstart, xgoal, n, r_rewire, r_goal
            # )
