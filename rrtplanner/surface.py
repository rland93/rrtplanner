import cvxpy as cp
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from .oggen import perlin_terrain


def xy_grid(xrange, yrange, shape):
    xr = np.linspace(xrange[0], xrange[1], shape[0])
    yr = np.linspace(yrange[0], yrange[1], shape[1])
    X, Y = np.meshgrid(xr, yr)
    return X, Y


class Surface(object):
    def __init__(self, X, Y, H, params):
        # check that shapes match
        if X.shape != Y.shape or X.shape != H.shape or Y.shape != H.shape:
            raise ValueError(
                f"X, Y, H must have the same shape. X: {X.shape}, Y: {Y.shape}, H: {H.shape}"
            )
        # check that problem parameters are present
        for p in ["minh", "gaph", "mindx", "mind2x"]:
            if p not in params:
                raise ValueError(f"{p} must be in params")

        self.X, self.Y, self.H = X, Y, None

        # set parameter H
        self.setH(H)
        self.S = cp.Variable(shape=X.shape, name="Surface")
        self.minh = params["minh"]
        self.gaph = params["gaph"]
        self.mindx = params["mindx"]
        self.mind2x = params["mind2x"]

        # problem
        self.problem = self.setUp()

    def setH(self, H):
        if self.H is None:
            self.H = cp.Parameter(shape=H.shape, name="Terrain Height", value=H)
        else:
            self.H.value = H

    def _diff(self, x_1, x, x1, h, k=1):
        if k == 1:
            """first derivative"""
            return (x1 - x_1) / (2 * h)
        elif k == 2:
            """second derivative"""
            return (x1 - 2 * x + x_1) / (h**2)

    def setUp(self, active_constr=["minh", "gaph", "mindx", "mind2x"]):
        objective = 0.0
        subjectto = []
        dh = self.X[0, 1] - self.X[0, 0]

        for constr in active_constr:
            if constr == "minh":
                c_minh = self.S >= self.minh
                subjectto += [c_minh]
            elif constr == "gaph":
                c_gaph = self.S - self.H >= self.gaph
                subjectto += [c_gaph]

        # # forward and backward differences

        fx_1 = self.S[:, 2:]  # x_i-1
        fx1 = self.S[:, :-2]  # x_i+1
        fx = self.S[:, 1:-1]  # x_i

        fy_1 = self.S[2:, :]  # y_i-1
        fy1 = self.S[:-2, :]  # y_i+1
        fy = self.S[1:-1, :]  # y_i

        # first diff
        c_dx = cp.abs(self._diff(fx_1, fx, fx1, dh, k=1)) <= self.mindx
        c_dy = cp.abs(self._diff(fy_1, fy, fy1, dh, k=1)) <= self.mindx

        # second diff
        c_d2x = cp.abs(self._diff(fx_1, fx, fx1, dh, k=2)) <= self.mind2x
        c_d2y = cp.abs(self._diff(fy_1, fy, fy1, dh, k=2)) <= self.mind2x

        subjectto += [c_dx, c_dy, c_d2x, c_d2y]
        objective = cp.sum_squares(self.S)

        return cp.Problem(cp.Minimize(objective), subjectto)

    def solve(self, verbose=False, solver="ECOS"):
        self.problem.solve(verbose=verbose, solver=solver)
        return self.S.value


def example_terrain(xmax, ymax, cols, rows):
    X, Y = xy_grid([0, xmax], [0, ymax], [cols, rows])
    H = perlin_terrain(cols, rows, scale=1.0)
    H *= perlin_terrain(cols, rows, scale=2.0)
    H *= perlin_terrain(cols, rows, scale=4.0)
    H = apply_ridge(X, H, steep=5.0, width=1.0, fn="bell")
    return X, Y, H


def apply_ridge(X, H, steep, width, fn="arctan"):
    """
    Apply a "ridge" function for terrain generation.
    """
    mid = np.ptp(X) / 2.0
    if fn == "arctan":
        for i in range(X.shape[1]):
            t1 = np.arctan(steep * (X[:, i] - mid) - width)
            t2 = np.arctan(steep * (X[:, i] - mid) + width)
            H[:, i] *= -(t1 - t2) / (2.0 * np.arctan(width))
    elif fn == "bell":
        for i in range(X.shape[1]):
            H[:, i] *= np.exp(-((X[:, i] - mid) ** 2) / (2.0 * steep**2))
    return H


if __name__ == "__main__":
    from oggen import perlin_terrain
    from plots import plot_surface
    from matplotlib.animation import FuncAnimation

    xmax, ymax = 80.0, 80.0
    cols, rows = 50, 40
    zsq = 0.5

    params = {
        "minh": 5.0,
        "gaph": 4.0,
        "mindx": 1.5,
        "mind2x": 0.1,
    }

    X, Y, H = example_terrain(xmax, ymax, cols, rows)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_surface(ax, X, Y, H, zsquash=zsq, wireframe=False, cmap="gist_earth")
    plt.show()

    surf = Surface(X, Y, H, params)
    S = surf.solve(verbose=True)

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(3, 3)

    ax0 = fig.add_subplot(gs[:2, :], projection="3d")
    ax1 = fig.add_subplot(gs[2, 0])
    ax2 = fig.add_subplot(gs[2, 1])

    ax1.contour(X, Y, S, levels=12, cmap="bone")
    plot_surface(ax0, X, Y, H, zsquash=zsq, wireframe=False, cmap="gist_earth")
    plot_surface(ax0, X, Y, S, zsquash=zsq, wireframe=True, cmap="bone")
    plt.show()
