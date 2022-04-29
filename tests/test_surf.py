from rrtplanner import surface, oggen, plots
import pytest
import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import pyplot as plt


@pytest.fixture
def og_terrain():
    return oggen.perlin_terrain(w=100, h=100, scale=10)


@pytest.fixture
def xygrid():
    return surface.xy_grid(xrange=(0, 1.0), yrange=(0, 1.0), shape=(5, 5))


def test_get_xy_grid():
    X, Y = surface.xy_grid(xrange=(0, 1.0), yrange=(0, 1.0), shape=(100, 100))
    assert X.shape == (100, 100)
    assert Y.shape == (100, 100)
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)


def test_plot_surf(xygrid):
    fig2d = plt.figure()
    ax2d = fig2d.add_subplot(111)

    X, Y = xygrid
    S = np.zeros(shape=X.shape, dtype=X.dtype)
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection="3d")

    with pytest.raises(TypeError) as excinfo:
        plots.plot_surface(ax2d, X, Y, S)
    plots.plot_surface(ax3d, X, Y, S)
