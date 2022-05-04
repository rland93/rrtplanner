if __name__ == "__main__":
    from rrtplanner import surface, plots, oggen
    import matplotlib.pyplot as plt

    xmax, ymax = 80.0, 80.0
    cols, rows = 60, 60
    zsq = 0.5

    params = {
        "minh": 5.0,
        "gaph": 4.0,
        "mindx": 1.5,
        "mind2x": 0.1,
    }

    X, Y, H = oggen.example_terrain(xmax, ymax, cols, rows)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plots.plot_surface(ax, X, Y, H, zsquash=zsq, wireframe=False, cmap="gist_earth")
    plt.show()

    surf = surface.Surface(X, Y, H, params)
    S = surf.solve(verbose=True)

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(3, 3)

    ax0 = fig.add_subplot(gs[:2, :], projection="3d")
    ax1 = fig.add_subplot(gs[2, 0])
    ax2 = fig.add_subplot(gs[2, 1])

    ax1.contour(X, Y, S, levels=12, cmap="bone")
    plots.plot_surface(ax0, X, Y, H, zsquash=zsq, wireframe=False, cmap="gist_earth")
    plots.plot_surface(ax0, X, Y, S, zsquash=zsq, wireframe=True, cmap="bone")
    plt.show()

    # now we can do a runtime analysis
    import time
    import numpy as np

    times = []
    times_warm_start = []
    for run in range(10):
        print(f"run: {run}")
        # update H
        _, _, newH = surface.example_terrain(xmax, ymax, cols, rows)
        surf.H.value = newH
        surf.solve(warm_start=False)
        times.append(surf.problem._solve_time)

    for run in range(10):
        print(f"run: {run}")
        # update H
        _, _, newH = surface.example_terrain(xmax, ymax, cols, rows)
        surf.H.value = newH
        surf.solve(warm_start=True)
        times_warm_start.append(surf.problem._solve_time)

    times = np.array(times)
    times_warm_start = np.array(times_warm_start)
    print("Average runtime (no warm start): {:.3f}s".format(times.mean()))
    print("Standard Deviation (no warm start): {:.3f}s".format(times.std()))
    print("Average runtime (warm start): {:.3f}s".format(times_warm_start.mean()))
    print("Standard Deviation (warm start): {:.3f}s".format(times_warm_start.std()))
