import cvxpy as cp
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from .oggen import perlin_terrain


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

    def _diff(
        self,
        x_1: cp.Variable,
        x: cp.Variable,
        x1: cp.Variable,
        h: float,
        k: int = 1,
    ):
        if k == 1:
            """first derivative"""
            return (x1 - x_1) / (2 * h)
        elif k == 2:
            """second derivative"""
            return (x1 - 2 * x + x_1) / (h**2)
        else:
            raise ValueError(f"k must be 1 or 2. k={k}")

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
        objective = cp.sum(self.S)

        problem = cp.Problem(cp.Minimize(objective), subjectto)
        return problem

    def solve(self, verbose=False, solver=cp.ECOS, warm_start=False):
        self.problem.solve(verbose=verbose, solver=solver, warm_start=warm_start)
        return self.S.value
