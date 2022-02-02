from rrtpp import rrt
import numpy as np
import pytest
from rrtpp import world_gen


point_dtypes = (
    int,
    float,
    np.uint32,
    np.uint64,
    np.int32,
    np.int64,
    np.float32,
    np.float64,
)


def world_param_generator():
    for dtype in point_dtypes:
        for world_topo in ("empty", "square", "empty"):
            for w in [43, 100]:
                for h in [43, 100]:
                    yield world_topo, dtype, w, h


@pytest.fixture(params=world_param_generator())
def make_world(request):
    world_topo, dtype, w, h = request.param
    if world_topo == "empty":
        empty_world = np.zeros((w, h), dtype=dtype)
        return empty_world
    elif world_topo == "square":
        square_world = np.zeros((w, h), dtype=dtype)
        square_world[w // 4 : 3 * w // 4, h // 4 : 3 * h // 4] = 1
        return square_world
    elif world_topo == "impossible":
        impossible_world = np.zeros((w, h), dtype=dtype)
        impossible_world[int(w / 2) : int(w / 2) + 1] = 1.0
        return impossible_world


def get_random_point(dtype):
    if any(
        (
            dtype == int,
            dtype == np.uint32,
            dtype == np.uint64,
            dtype == np.int32,
            dtype == np.int64,
        )
    ):
        x1 = np.random.randint(0, 10000)
        x2 = np.random.randint(0, 10000)
        low = min(x1, x2)
        high = max(x1, x2)
        point = np.random.randint(low=low, high=high, size=(2,)).astype(dtype)
    elif any((dtype == float, dtype == np.float32, dtype == np.float64)):
        x1 = np.random.uniform(0, 10000)
        x2 = np.random.uniform(0, 10000)
        low = min(x1, x2)
        high = max(x1, x2)
        point = np.random.uniform(low=low, high=high, size=(2,)).astype(dtype)
    return point


@pytest.mark.parametrize("dtype", point_dtypes)
def test_r2norm(dtype):
    point = get_random_point(dtype)
    assert np.isclose(rrt.r2norm(point), np.linalg.norm(point))


@pytest.fixture(params=[25, 100, 400, 1000])
def getrrtobj(make_world, request):
    n = request.param
    return rrt.RRT(make_world, n)


def random_2_individual_points(rrtobj):
    x1 = np.random.randint(0, rrtobj.world.shape[0])
    x2 = np.random.randint(0, rrtobj.world.shape[0])
    y1 = np.random.randint(0, rrtobj.world.shape[1])
    y2 = np.random.randint(0, rrtobj.world.shape[1])
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    return p1, p2


def random_points_array(rrtobj, n):
    pointsx = np.random.randint(0, rrtobj.world.shape[0], size=(n,))
    pointsy = np.random.randint(0, rrtobj.world.shape[0], size=(n,))
    points = np.stack([pointsx, pointsy], -1)
    return points


def test_collision(getrrtobj):
    rrtobj = getrrtobj
    p1, p2 = random_2_individual_points(rrtobj)
    rrtobj.collisionfree(rrtobj.world, p1, p2)


def test_near(getrrtobj):
    rrtobj = getrrtobj
    p1, _ = random_2_individual_points(rrtobj)
    points = random_points_array(rrtobj, 10)
    rrtobj.near(points, p1)


def test_within(getrrtobj):
    rrtobj = getrrtobj
    p1, p2 = random_2_individual_points(rrtobj)
    points = random_points_array(rrtobj, 10)
    rrtobj.within(points, p1, 10)

    points_ = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    p1_ = np.array([0.5, 0.5])

    assert rrtobj.within(points_, p1_, 1.0).shape[0] == 4
