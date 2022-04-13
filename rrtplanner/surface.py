import cvxpy as cp
import numpy as np


def getsurface(
    X, Y, zterrain, zgap, zmin, xystep, vmin, amin, waypoints=None, waypointcost=None
):
    """Get grid from occupancygrid og"""
    S = cp.Variable(shape=zterrain.shape)

    cost, constraints = 0.0, []
    # constraints
    constraints.append(S >= zmin)  # min height
    constraints.append(S - zterrain >= zgap)  # gap

    # first deriv constraints
    dx = cp.abs(cp.diff(S, axis=0))
    dy = cp.abs(cp.diff(S, axis=1))
    dxc = dx <= vmin * xystep
    dyc = dy <= vmin * xystep
    constraints.append(dxc)
    constraints.append(dyc)

    # second deriv constraints
    ddx = cp.abs(cp.diff(S, axis=0, k=2))
    ddy = cp.abs(cp.diff(S, axis=1, k=2))
    ddxc = ddx <= amin * xystep**2
    ddyc = ddy <= amin * xystep**2
    constraints.append(ddxc)
    constraints.append(ddyc)

    # waypoint constraints
    if waypoints is not None:
        for wp in waypoints:
            # get quadrilateral
            wpy_i = np.argsort(np.abs(X - wp[0]), axis=1)[0][:4]
            wpx_i = np.argsort(np.abs(Y - wp[1]), axis=0)[:4][:, 0]
            # average of quadrilateral
            cost += cp.sum_squares(S[wpx_i, wpy_i] - wp[2]) * waypointcost / 4.0

    # per-grid point cost
    cost += cp.sum(S) / (S.shape[0] * S.shape[1])

    # solve
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver="ECOS", verbose=True)

    # log waypoint error
    if waypoints is not None:
        for wp in waypoints:
            i = np.argsort(np.abs(X - wp[0]), axis=1)[0][:4]
            j = np.argsort(np.abs(Y - wp[1]), axis=0)[:4][:, 0]
            wp_error = np.mean(np.abs(S.value[i, j] - wp[2]))
            print(wp_error)

    # get surface
    return S.value


if __name__ == "__main__":
    import os
    from oggen import perlin_terrain
    from plots import plot_3D_terrain
    from random import uniform
    from matplotlib import pyplot as plt

    # z squash
    ZSQ = 0.33

    # terrain
    tcmap = "gist_earth"

    figsz = (6, 6)

    w, h = 160, 160
    ridgemid = w / 2.0
    ridgewid = w / 5.0
    step = 1.0
    z_gap = 15.0
    z_min = 25.0

    dx = 1.4
    ddx = 0.10

    nwaypoints = 2

    # get some random terrain
    alpha = 80  # high freq scale
    beta = 50  # low freq scale
    X, Y = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
    Z = perlin_terrain(w, h, scale=1.5) * alpha + perlin_terrain(w, h, scale=0.3) * beta

    ridge = np.empty((w, h))
    ridgex = X[0, :]

    def ridge_fn(x, mid, width):
        return np.exp(-(((x - mid) / width) ** 2))

    ridgey = ridge_fn(ridgex, ridgemid, ridgewid)
    for i in range(0, w):
        ridge[i, :] = ridgey

    Z *= ridge

    terrain_fig_test = plt.figure()
    ax = terrain_fig_test.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap=tcmap)
    plt.show()

    fig0 = plt.figure(tight_layout=True, figsize=figsz)
    ax0 = fig0.add_subplot(111, projection="3d")
    ax0 = plot_3D_terrain(ax0, X, Y, Z, zsquash=ZSQ, cmap=tcmap)
    ax0 = plot_3D_terrain(
        ax0, X, Y, Z + z_gap, zsquash=ZSQ, wireframe=True, cmap="bone"
    )
    ax0.set_title(r"Naive: $z_{sheet} = z_{terrain} + z_{gap}$")

    # 1st Derivative Only
    S1 = getsurface(X, Y, Z, z_gap, z_min, step, dx, 1000.0)
    fig1 = plt.figure(tight_layout=True, figsize=figsz)
    ax1 = fig1.add_subplot(111, projection="3d")
    ax1 = plot_3D_terrain(ax1, X, Y, Z, zsquash=ZSQ, cmap=tcmap)
    ax1 = plot_3D_terrain(ax1, X, Y, S1, zsquash=ZSQ, wireframe=True, cmap="bone")
    ax1.set_title(r"Optimal: Gap and 1st Deriv")

    # 2nd Derivative + 1st Derivative
    S2 = getsurface(X, Y, Z, z_gap, z_min, step, dx, ddx)
    fig2 = plt.figure(tight_layout=True, figsize=figsz)
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2 = plot_3D_terrain(ax2, X, Y, Z, zsquash=ZSQ, cmap=tcmap)
    ax2 = plot_3D_terrain(ax2, X, Y, S2, zsquash=ZSQ, wireframe=True, cmap="bone")
    ax2.set_title(r"Optimal: Gap, 1st, 2nd Deriv Constraints")

    # 1st, 2nd, and Waypoints
    fig3 = plt.figure(tight_layout=True, figsize=figsz)
    ax3 = fig3.add_subplot(111, projection="3d")

    waypoints = []
    waypoints.append([6 * w // 7, 6 * h // 7, max(alpha, beta) + 40.0])
    waypoints.append([w // 7, h // 7, max(alpha, beta) + 25.0])
    waypoints = np.array(waypoints)

    S3 = getsurface(
        X, Y, Z, z_gap, z_min, step, dx, ddx, waypoints=waypoints, waypointcost=1e3
    )

    ax3 = plot_3D_terrain(ax3, X, Y, Z, zsquash=ZSQ, cmap=tcmap)
    ax3 = plot_3D_terrain(ax3, X, Y, S3, zsquash=ZSQ, wireframe=True, cmap="bone")
    ax3.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], c="r", s=40)
    ax3.set_title("Optimal: Gap, 1st, 2nd Deriv Constraints, Waypoint Costs")

    for a in [ax0, ax1, ax2, ax3]:
        a.set_xticks([])
        a.set_yticks([])
        a.set_zticks([])

    plt.show()
    if not os.path.exists("./figure_output"):
        os.makedirs("./figure_output")
    fig0.savefig("./figure_output/scratch.png", dpi=300)
    for i, f in enumerate([fig0, fig1, fig2, fig3]):
        print("Saving figure {}/4".format(i))
        f.savefig("./figure_output/{}.png".format(i), dpi=300)
