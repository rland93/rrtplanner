import cvxpy as cp
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


def xy_grid(xrange, yrange, shape):
    xr = np.linspace(xrange[0], xrange[1], shape[0])
    yr = np.linspace(yrange[0], yrange[1], shape[1])
    X, Y = np.meshgrid(xr, yr)
    return X, Y


if __name__ == "__main__":

    from oggen import perlin_terrain
    from plots import plot_surface
    from time import time

    cols, rows = 64, 64
    zsq = 0.5

    X, Y = xy_grid(xrange=(0, 10.0), yrange=(0, 10.0), shape=(rows, cols))

    H = cp.Parameter(
        shape=X.shape,
        name="Terrain",
        value=perlin_terrain(w=cols, h=rows, scale=2) * 10.0,
    )
    S = cp.Variable(shape=X.shape, name="Surface")

    # optimization variables
    minh, gaph, mindx, mind2x = 5.0, 2.0, 0.5, 0.1

    h = X[0, 1] - X[0, 0]
    objective = 0.0
    subjectto = []

    c_minh = S >= minh
    c_gaph = S - H >= gaph
    subjectto += [c_minh, c_gaph]

    # Forward/backward differences
    backwardx = S[:, 2:]
    forwardx = S[:, :-2]
    backwardy = S[2:, :]
    forwardy = S[:-2, :]

    # First Derivative Constraint
    dx = (forwardx - backwardx) / (2 * h)
    dy = (forwardy - backwardy) / (2 * h)
    c_dx = cp.abs(dx) <= mindx
    c_dy = cp.abs(dy) <= mindx
    subjectto += [c_dx, c_dy]

    # Second Derivative Constraint
    d2x = (forwardx - 2 * S[:, 1:-1] + backwardx) / (h**2)
    d2y = (forwardy - 2 * S[1:-1, :] + backwardy) / (h**2)
    c_d2x = cp.abs(d2x) <= mind2x
    c_d2y = cp.abs(d2y) <= mind2x
    subjectto += [c_d2x, c_d2y]

    objective += cp.sum_squares(S)
    prob = cp.Problem(cp.Minimize(objective), subjectto)
    prob.solve(solver="ECOS", verbose=True)

    n = 6
    times, times_ws = [], []
    for i in range(n):
        print(f"Naive Solve {i}")
        t0 = time()
        H.value = perlin_terrain(w=cols, h=rows, scale=2) * 10.0
        objective = 0.0
        prob.solve(solver="ECOS", verbose=False)
        t1 = time()
        times.append(t1 - t0)

    for i in range(n):
        print(f"Warm Start Solve {i}")
        t0 = time()
        H.value = perlin_terrain(w=cols, h=rows, scale=2) * 10.0
        objective = 0.0
        prob.solve(solver="ECOS", verbose=False, warm_start=True)
        t1 = time()
        times_ws.append(t1 - t0)

    times = np.array(times)
    times_ws = np.array(times_ws)
    print(f"{n} iterations: {times.mean():.3f}s, {times_ws.mean():.3f}s")
    print(f"{n} iterations: {times.std():.3f}s, {times_ws.std():.3f}s")

    # fig = plt.figure()
    # ax0 = fig.add_subplot(121)
    # ax1 = fig.add_subplot(122, projection="3d")
    # ax0.imshow(S, cmap=cm.viridis)
    # plot_surface(ax1, X, Y, H, zsquash=zsq, wireframe=False, cmap="gist_earth")
    # plot_surface(ax1, X, Y, S, zsquash=zsq, wireframe=True, cmap="bone")
    # plt.show()
