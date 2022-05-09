import cvxpy as cp
import numpy as np
from inspect import signature
from .rrt import RRTStar
from collections import defaultdict
from tqdm import tqdm


class SurfaceBase(object):
    """base class for Surface object"""

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        H: np.ndarray,
    ):
        # check that shapes match
        if X.shape != Y.shape or X.shape != H.shape or Y.shape != H.shape:
            raise ValueError(
                f"X, Y, H must have the same shape. X: {X.shape}, Y: {Y.shape}, H: {H.shape}"
            )

        self.X, self.Y, self.H = X, Y, None

        # set parameter H
        self.setH(H)
        self.S = cp.Variable(shape=H.shape, name="Surface")

        # problem
        self.problem = None

        # objectives
        self.objectives = {}

    def check_setup(self):
        """check that setup method has use_parameters kwarg."""
        # check that use_parameters kwarg
        for p in signature(self.setup).parameters.values():
            if p.kind == p.POSITIONAL_OR_KEYWORD and p.name == "use_parameters":
                return
        raise ValueError("setup() must have a use_parameters kwarg!")

    def setH(self, H: np.ndarray) -> None:
        """Set the terrain height parameter. If there is no terrain height parameter belonging to this method's object, this method will create one.

        Parameters
        ----------
        H : np.ndarray
            The terrain height at each grid point.
        """
        if self.H is None:
            self.H = cp.Parameter(shape=H.shape, name="Terrain Height", value=H)
        else:
            self.H.value = H

    def _diff(self, x: cp.Variable, h: float, k: int = 1):
        xbak = x[:, 2:]
        xfor = x[:, :-2]
        xmid = x[:, 1:-1]
        ybak = x[2:, :]
        yfor = x[:-2, :]
        ymid = x[1:-1, :]
        if k == 1:
            """first derivative"""
            dx = (xfor - xbak) / (2 * h)
            dy = (yfor - ybak) / (2 * h)
            return dx, dy
        elif k == 2:
            """second derivative"""
            d2x = (xfor - 2 * xmid + xbak) / (h**2)
            d2y = (yfor - 2 * ymid + ybak) / (h**2)
            return d2x, d2y
        else:
            raise ValueError(f"k must be 1 or 2. k={k}")

    def solve(self, verbose=False, solver=cp.ECOS, warm_start=False):
        """Solve the optimization problem. `warm_start` will use previous solutions to speed up the optimization.


        Parameters
        ----------
        verbose : bool, optional
            pass verbose to the solver, by default False
        solver : str, optional
            which solver to use, by default cp.ECOS
        warm_start : bool, optional
            whether to use warm start, by default False

        Returns
        -------
        np.ndarray
            The value of the sheet.
        """
        if self.problem is not None:
            self.problem.solve(verbose=verbose, solver=solver, warm_start=warm_start)
            return self.S.value
        else:
            raise ValueError("You must first call setup() to set problem parameters.")

    def setup():
        raise NotImplementedError(
            "This is the base class for Surface. You must call setup() on a subclass!"
        )


class SurfaceWaypoints(SurfaceBase):
    def __init__(self, X, Y, H):
        super().__init__(X, Y, H)
        self.check_setup()

    def setup(
        self,
        gaph: float,
        maxdx: float,
        maxd2x: float,
        waypoints: np.ndarray,
        d2x_cost: float = 1.0,
        waypointcost: float = 1.0,
        heightcost: float = 1.0,
        use_parameters: bool = True,
    ):
        """Set up a curvature-penalty problem. This problem attempts to find a surface which passes as close to the waypoints as possible, while minimizing the curvature of the curve. No height penalty is applied.

        Parameters
        ----------
        gaph : float
            Minimum distance between sheet and terrain
        maxdx : float
            Maximum surface slope
        maxd2x : float
            Maximum surface curvature
        waypoints : np.ndarray
            Mx3 array of M waypoints, each with a height.
        waypointcost : float
            Cost incurred by missing waypoints. 1.0 is default.
        heightcost : float
            Cost incurred by altitude.
        use_parameters : bool, by default True
            Whether to use cp.Parameter objects when setting terrain height
        """
        objective, subjectto = 0.0, []
        dh = self.X[0, 1] - self.X[0, 0]
        # gap height constraint
        if use_parameters:
            H = self.H
        else:
            H = self.H.value
        c_gaph = self.S - H >= gaph
        # first diff
        dx, dy = self._diff(self.S, dh, k=1)
        c_dx, c_dy = cp.abs(dx) <= maxdx, cp.abs(dy) <= maxdx
        # second diff
        d2x, d2y = self._diff(self.S, dh, k=2)
        c_d2x, c_d2y = cp.abs(d2x) <= maxd2x, cp.abs(d2y) <= maxd2x
        subjectto += [c_dx, c_dy, c_d2x, c_d2y, c_gaph]
        # waypoint penalty
        waypoint_penalty = 0.0
        for waypoint in waypoints:
            closest_idxs_x = np.argsort(np.abs(self.X - waypoint[0]), axis=1)[0, :4]
            closest_idxs_y = np.argsort(np.abs(self.Y - waypoint[1]), axis=0)[:4, 0]
            # average altitude of closest quadrilateral.
            S_closest = cp.sum(self.S[closest_idxs_y, closest_idxs_x]) / 4.0
            # waypoint penalty is squared difference between waypoint alt. and quad alt.
            waypoint_penalty += cp.square((S_closest - waypoint[2]))
        # average derivative for each point
        deriv_penalty = (cp.sum_squares(d2x) + cp.sum_squares(d2y)) / (
            self.S.shape[0] * self.S.shape[1]
        )

        # height penalty
        if heightcost is not None:
            # average height
            height_penalty = cp.sum(self.S) / (self.S.shape[0] * self.S.shape[1])
        else:
            height_penalty = 0.0
        objective = (
            waypoint_penalty * waypointcost
            + d2x_cost * deriv_penalty
            + height_penalty * heightcost
        )
        problem = cp.Problem(cp.Minimize(objective), subjectto)
        self.problem = problem
        # this is the easiest way to merge two dicts lmao
        self.objectives = dict(
            self.objectives,
            **{
                "waypoint": waypoint_penalty,
                "deriv": deriv_penalty,
                "height": height_penalty,
            },
        )
        print(f"heightcost = {heightcost}")
        print(f"waypointcost = {waypointcost}")
        print(f"d2x_cost = {d2x_cost}")
        return problem


class SurfaceLowHeight(SurfaceBase):
    def __init__(self, X, Y, H):
        super().__init__(X, Y, H)

    def setup(
        self, minh: float, gaph: float, maxdx: float, maxd2x: float, use_parameters=True
    ) -> cp.Problem:
        """Set up a height-penalty problem. The problem is to minimize the surface height, while obeying the constraints:

        - The surface height is greater than the terrain height plus `gaph`.
        - The surface height is greater than the minimum height.
        - The first derivative of the surface (dh/dx) is less than `maxdx`.
        - The second derivative of the surface (d2h/dx2) is less than `maxd2x`.

        Parameters
        ----------
        minh : float
            Minimum height of the surface, independent of terrain.
        gaph : float
            Minimum height between the surface and the terrain.
        maxdx : float
            Maximum first derivative (slope) of the surface.
        maxd2x : float
            Maximum second derivative (curvature) of the surface.
        use_parameters : bool, by default True
            Whether to use cp.Parameter objects when setting terrain height

        Returns
        -------
        cp.Problem
            Problem to be solved.
        """
        objective = 0.0
        subjectto = []
        dh = self.X[0, 1] - self.X[0, 0]
        # min height constraint
        c_minh = self.S >= minh
        if use_parameters:
            H = self.H
        else:
            H = self.H.value
        c_gaph = self.S - H >= gaph
        # first diff
        dx, dy = self._diff(self.S, dh, k=1)
        c_dx, c_dy = cp.abs(dx) <= maxdx, cp.abs(dy) <= maxdx
        # second diff
        d2x, d2y = self._diff(self.S, dh, k=2)
        c_d2x, c_d2y = cp.abs(d2x) <= maxd2x, cp.abs(d2y) <= maxd2x
        # add constraints
        subjectto += [c_gaph, c_minh, c_dx, c_dy, c_d2x, c_d2y]
        # objective function
        objective = cp.sum(self.S)
        problem = cp.Problem(cp.Minimize(objective), subjectto)
        self.problem = problem
        self.objectives = {"Sum height": objective}
        print(f"Sum Height = {objective}")
        return problem


def generate_example_waypoints(
    X: np.ndarray,
    Y: np.ndarray,
    H: np.ndarray,
    nwaypoints: int,
    above_range: tuple,
):
    """Generate a set of example waypoints for a given terrain. These waypoints are not necessarily tractable, depending on the constraints given to the sheet (i.e., a line connecting two waypoints could be impossible for a given slope constraint). But, the example waypoints will always lie above the terrain, in the band specified by `above_range`.

    Parameters
    ----------
    X : np.ndarray
        X points (MxN)
    Y : np.ndarray
        Y points (MxN)
    H : np.ndarray
        Terrain points (MxN)
    nwaypoints : int
        No. of waypoints to generate
    above_range : tuple
        Range above terrain the waypoints should be placed.


    Returns
    -------
    np.ndarray
        (Mx3) array of waypoints.
    """
    waypoints = np.full((nwaypoints, 3), fill_value=float("inf"))
    for i in range(nwaypoints):
        wayp_i = np.random.randint(0, X.shape[0])
        wayp_j = np.random.randint(0, Y.shape[1])
        wayp_h = H[wayp_i, wayp_j] + np.random.uniform(*above_range)
        waypoints[i] = [X[wayp_i, wayp_j], Y[wayp_i, wayp_j], wayp_h]
    return waypoints
