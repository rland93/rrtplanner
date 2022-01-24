from world_gen import make_terrain
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize
from math import ceil
import cvxpy as cp
import numpy as np
import numba as nb


def plot_surface(
    ax,
    X,
    Y,
    H,
    cmap="gist_earth",
    wireframe=False,
    set_aspect_to_data=False,
    cmap_pad=0.1,
):

    ax.set_proj_type("ortho")
    pad = (H.max() - H.min()) * cmap_pad
    norm = Normalize(vmin=H.min() - pad, vmax=H.max() + pad)
    colors = cm.get_cmap(cmap)(norm(H))
    if wireframe:
        S = ax.plot_surface(
            X,
            Y,
            H,
            zorder=2,
            linewidths=0.5,
            shade=False,
            facecolors=colors,
        )
        S.set_facecolor([0, 0, 0, 0])
    else:
        S = ax.plot_surface(X, Y, H, zorder=2, facecolors=colors, alpha=1.0)

    if set_aspect_to_data:
        ax.set_box_aspect(
            [
                X.max() - X.min(),
                Y.max() - Y.min(),
                H.max() - H.min(),
            ]
        )
    return S


def get_surface(
    X,
    Y,
    Hterrain,
    buffer,
    min_h,
    grid_size,
    max_dh,
    max_d2h,
    verbose=False,
    solver=None,
):
    constraints = []
    cost = 0

    # new h is a free variable with same shape as
    surface = cp.Variable(shape=X.shape, name="surface")

    # buffer constraint
    buffer_constraint = surface - Hterrain >= buffer
    constraints.append(buffer_constraint)

    # minimum height constraint
    min_h_constraint = surface >= min_h
    constraints.append(min_h_constraint)

    # 1st partial (climb rate)
    dhx = cp.diff(surface, 1, axis=0)
    dhy = cp.diff(surface, 1, axis=1)
    # climb rate constraint
    climb_rate_constraint_x = cp.abs(dhx) <= max_dh * grid_size
    climb_rate_constraint_y = cp.abs(dhy) <= max_dh * grid_size
    constraints.append(climb_rate_constraint_x)
    constraints.append(climb_rate_constraint_y)

    # 2nd partial (change in climb rate)
    d2hx = cp.diff(surface, 2, axis=0)
    d2hy = cp.diff(surface, 2, axis=1)

    change_climb_rate_constraint_x = cp.abs(d2hx) <= max_d2h * grid_size * 2
    change_climb_rate_constraint_y = cp.abs(d2hy) <= max_d2h * grid_size * 2
    constraints.append(change_climb_rate_constraint_x)
    constraints.append(change_climb_rate_constraint_y)

    # cost += cp.sum(cp.abs(dhx)) * climb_cost
    # cost += cp.sum(cp.abs(dhy)) * climb_cost
    # cost += cp.sum(cp.abs(d2hx)) * climb_rate_cost
    # cost += cp.sum(cp.abs(d2hy)) * climb_rate_cost

    # lowest possible
    cost += cp.sum(surface)
    problem = cp.Problem(cp.Minimize(cost), constraints)

    # solve problem
    if solver is not None:
        problem.solve(verbose=verbose, solver=solver)
    else:
        problem.solve(verbose=verbose)

    # return solved value
    return surface.value


@nb.njit()
def line_idx(x0, x1, y0, y1):
    """integer line algorithm"""
    d1 = max(x0, x1) - min(x0, x1)
    d2 = max(y0, y1) - min(y0, y1)
    # preallocate
    line = np.empty((max(d1, d2) + 1, 2), dtype=np.int64)
    line[0] = [x0, y0]
    i = 1

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
        if x0 == x1 and y0 == y1:
            line[i] = (x0, y0)
            return line
        else:
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
            line[i] = (x0, y0)
            i += 1


if __name__ == "__main__":
    from world_gen import get_rand_start_end, make_terrain
    from scipy.interpolate import RectBivariateSpline
    from math import sqrt
    import numba as nb

    w, h = 120, 80
    X, Y, Hterrain = make_terrain((w, h), (6, 4), cutoff=True, thresh=0.34, height=4)
    # open space on the borders
    Hterrain[:10, :] = 0
    Hterrain[-10:, :] = 0
    # get start goal
    xstart, xgoal = np.array([int(5), int(h / 2)]), np.array([int(w - 5), int(h / 2)])

    # scale aspect to terrain, but exaggerated
    z_exaggerate = 1.0
    terrain_aspect = [
        X.max() - X.min(),
        Y.max() - Y.min(),
        (Hterrain.max() - Hterrain.min()) * z_exaggerate,
    ]

    surface = get_surface(
        X,
        Y,
        Hterrain,
        2.5,
        0.5,
        1.0,
        np.tan(1.0),
        0.04,
    )

    from rrt import RRTStar

    cost_hscale = 1e2

    @nb.njit(fastmath=True)
    def r2norm(x):
        return sqrt(x[0] * x[0] + x[1] * x[1])

    @nb.njit(fastmath=True)
    def costfn(vcosts, points, v, x) -> float:
        if vcosts[v] < 1e10:
            # get integer representation of points
            ax, ay = points[v]
            bx, by = x
            # get discrete points from surface
            line = line_idx(ax, bx, ay, by)
            # get h-value of surface at each idx point on line
            # we need to unpack these if they are in compiled func
            linex, liney = line[:, 0], line[:, 1]

            # get cost of each line point
            cost = np.empty((linex.shape[0],), dtype=np.float64)
            for i in np.arange(linex.shape[0]):
                lx, ly = linex[i], liney[i]
                cost[i] = surface[lx, ly]

            # average h-val in cells
            cost = np.sum(cost) / cost.shape[0]
            # integrate average cost over line distance
            # then multiply by scaling factor.
            cost *= r2norm(points[v] - x) * cost_hscale
            # finally, we add the r2 distace, so that in very
            # low-cost (areas where cost ~= 0) we still get min
            # distance rewirings.
            cost += r2norm(points[v] - x)
            # add r2 distance
            return vcosts[v] + cost
        else:
            return np.inf

    rrts = RRTStar(Hterrain, n=4000, r_rewire=w, costfn=costfn)
    T, vgoal = rrts.make(xstart, xgoal)

    rrts_c = RRTStar(Hterrain, n=4000, r_rewire=w)
    T_c, vgoal_c = rrts_c.make(xstart, xgoal)

    from plots import plot_path, plot_rrt_lines, plot_start_goal

    fig2, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), tight_layout=True)
    for ij in np.ndindex(ax.shape):
        plot_start_goal(ax[ij], xstart, xgoal)
        ax[ij].imshow(Hterrain.T, cmap="binary", origin="lower")
        i, j = ij
        # first row plots
        if i == 0:
            G, vg = T_c, vgoal_c
            rrts_ = rrts
        # second row plots
        elif i == 1:
            G, vg = T, vgoal
            rrts_ = rrts_c
            if j == 1 or j == 2:
                ax[ij].contour(
                    X,
                    Y,
                    surface,
                    levels=9,
                    cmap="inferno",
                    linewidths=0.5,
                    antialiased=True,
                )
        if j == 1 or j == 2:
            path = rrts_.path_points(G, rrts_.route2gv(G, vg))
            plot_path(ax[ij], path)
        if j == 2:
            plot_rrt_lines(ax[ij], G, cmap="BuRd")

        title = (
            r"RRT* ($\mathbb{R}^2$ Distance Heuristic)"
            if i == 0
            else r"RRT* (Optimal Cost Heuristic)"
        )
        title += " "
        if j == 0:
            title += "Cspace Only"
        elif j == 1:
            title += "Path"
        elif j == 2:
            title += "Tree"
        ax[i, j].set_title(title)
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)
    fig2.savefig("rrt_surface.png", dpi=300)

    plt.show()
