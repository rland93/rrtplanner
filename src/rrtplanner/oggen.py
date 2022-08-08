import pyfastnoisesimd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def perlin_occupancygrid(
    w: int, h: int, thresh: float = 0.33, frames: int = None
) -> np.ndarray:
    """Make an occupancyGrid with thresholded Perlin Noise.

    Parameters
    ----------
    w : int
        width of the occupancyGrid
    h : int
        height of the occupancyGrid
    thresh : float, optional
        threshold of the noise. 0.1 produces an occupancyGrid that is mostly free space,
        0.9 produces an occupancyGrid that is mostly obstacle space. by default 0.3
    frames : int, optional
        number of "frames" of occupancygrid. If a number is passed, this will produce a
        3-dimensional occupancygrid, with the first dimension being time. This is
        designed to be used to simulate e.g. a dynamically, but smoothly changing
        occupancyGrid., by default None

    Returns
    -------
    np.ndarray
        occupancyGrid where 1 is occupied and 0 is free space.
    """
    # create noise
    noise = pyfastnoisesimd.Noise()
    noise.noiseType = pyfastnoisesimd.NoiseType(5)
    if frames is not None:
        xynoise = noise.genAsGrid(shape=[frames, w, h])
    else:
        xynoise = noise.genAsGrid(shape=[1, w, h])
        xynoise = np.squeeze(xynoise)
    # normalize to [0,1]
    xynoise -= xynoise.min()
    xynoise = xynoise / (xynoise.max() - xynoise.min())
    # threshold
    xynoise = np.where(xynoise >= thresh, 0, 1)
    return xynoise


def perlin_terrain(w: int, h: int, scale: int = 1, frames: int = None) -> np.ndarray:
    """Make a terrain with thresholded Perlin Noise.

    Parameters
    ----------
    w : int
        width of the terrain
    h : int
        height of the terrain
    frames : int, optional
        number of "frames" of terrain. If a number is passed, this will produce a
        3-dimensional terrain, with the first dimension being time. This is
        designed to be used to simulate e.g. a dynamically, but smoothly changing
        terrain., by default None

    Returns
    -------
    np.ndarray
        terrain array where value is terrain height (float between 0, 1).
    """
    # create noise
    noise = pyfastnoisesimd.Noise()
    noise.frequency = 0.01 * scale
    noise.noiseType = pyfastnoisesimd.NoiseType(5)
    if frames is not None:
        xynoise = noise.genAsGrid(shape=[frames, h, w])
    else:
        xynoise = noise.genAsGrid(shape=[1, h, w])
        xynoise = np.squeeze(xynoise)
    # normalize to [0,1]
    xynoise -= xynoise.min()
    xynoise = xynoise / (xynoise.max() - xynoise.min())
    return xynoise


def apply_ridge(X, H, steep, width, noise=5.0, fn="arctan"):
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
            hillcurve = np.exp(-((X[:, i] - mid) ** 2) / (2.0 * steep**2))
            H[:, i] *= hillcurve
    return H


def xy_grid(xrange, yrange, shape):
    xr = np.linspace(xrange[0], xrange[1], shape[0])
    yr = np.linspace(yrange[0], yrange[1], shape[1])
    X, Y = np.meshgrid(xr, yr)
    return X, Y


def example_terrain(xmax, ymax, hmax, cols, rows):
    X, Y = xy_grid([0, xmax], [0, ymax], [cols, rows])
    H = perlin_terrain(cols, rows, scale=1.0)
    H += 2.0 * perlin_terrain(cols, rows, scale=2.0)
    H *= 4.0 * perlin_terrain(cols, rows, scale=2.0)
    H = apply_ridge(X, H, steep=20.0, width=1.0, fn="bell")
    H *= float(hmax) / np.ptp(H)
    return X, Y, H


def corridor_og(shape, xposition=0.4, xwidth=0.1, yposition=0.5, opening=0.35):
    og = np.zeros(shape)
    xstart = xposition - xwidth / 2.0
    xend = xposition + xwidth / 2.0
    ystart = yposition - opening / 2.0
    yend = yposition + opening / 2.0
    # create "wall"
    og[:, int(xstart * shape[1]) : int(xend * shape[1])] = 1.0
    # create "hole" in the "wall"
    og[int(ystart * shape[0]) : int(yend * shape[0]), :] = 0.0
    return og


if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xynoise = perlin_occupancygrid(1000, 1000, frames=1000)
    im = ax.imshow(xynoise[0], cmap="gray_r", interpolation=None)

    def anim_f(n):
        im.set_data(xynoise[n])
        return (im,)

    animation = FuncAnimation(
        fig, anim_f, frames=xynoise.shape[0], interval=10, blit=True
    )
    plt.show()
