if __name__ == "__main__":
    from rrtplanner import surface, plots, oggen
    import os, pathlib

    WIDTH, HEIGHT, DEPTH = 100, 64, 35

    aspectyx = float(HEIGHT) / WIDTH
    aspectzx = float(DEPTH) / HEIGHT

    xmax, ymax = WIDTH, HEIGHT
    cols, rows = WIDTH, HEIGHT
    gaph, minh, maxdx, maxd2x = 2.5, 17.5, 0.95, 0.07
    X, Y, H = oggen.example_terrain(xmax=xmax, ymax=ymax, cols=cols, rows=rows)
    H *= float(DEPTH) / H.ptp()
    waypoints = surface.generate_example_waypoints(
        X,
        Y,
        H,
        2,
        (10.0, 35.0),
    )

    # "High" Surface
    surf1 = surface.SurfaceWaypoints(X, Y, H)
    surf1.setup(
        gaph,
        maxdx,
        maxd2x,
        waypoints,
        use_parameters=False,
        heightcost=1e1,
        waypointcost=1e2,
    )
    surf1.solve(verbose=True)
    for k, v in surf1.objectives.items():
        print(f"{k}: {v.value}")

    plots.generate_ensemble_plotly(
        X,
        Y,
        H,
        surf1.S.value,
        aspectyx=aspectyx,
        aspectzx=aspectzx,
        waypoints=waypoints,
        terrain_cmap="Dense_r",
        surface_cmap="Blues_r",
    )

    # "Low Surface"
    surf2 = surface.SurfaceLowHeight(X, Y, H)
    surf2.setup(
        minh,
        gaph,
        maxdx,
        maxd2x,
        use_parameters=False,
    )
    surf2.solve(verbose=True)
    for k, v in surf2.objectives.items():
        print(f"{k}: {v.value}")
    plots.generate_ensemble_plotly(
        X,
        Y,
        H,
        surf2.S.value,
        aspectyx=aspectyx,
        aspectzx=aspectzx,
        terrain_cmap="Dense_r",
        surface_cmap="Reds_r",
    )

    if False:
        os.makedirs("./plots", exist_ok=True)
        folder = pathlib.Path("./plots").resolve()
        try:
            highest = max(int(file.stem) for file in folder.glob("*.png")) + 1
        except ValueError:
            highest = 0
        location = str(folder / str(f"{highest}".zfill(2) + ".png"))
        print(f"saved file to {location}")
        fig.savefig(location, dpi=300)
