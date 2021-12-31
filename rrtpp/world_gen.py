from matplotlib.collections import PatchCollection
import numpy as np
from matplotlib.patches import Rectangle
import perlin_numpy
import matplotlib.pyplot as plt
from matplotlib import animation
from typing import Tuple


class ObstacleGenerator(object):
    def __init__(self, world, subdivide=False):
        self.world = world
        if subdivide:
            self.obstacles = self.generate_subdivisions(obs=True)

    def generate_subdivisions(self, obs=True):
        ax0 = 0
        ay0 = 0
        ax1 = self.world.shape[0]
        ay1 = self.world.shape[1]
        sw = []
        self.subdivide(ax0, ay0, ax1, ay1, sw=sw, obs=obs)
        return sw

    def subdivide(self, x0, y0, x1, y1, sw=[], obs=True):
        """Recursive quadtree subdivision algorithm"""
        if x1 - x0 == 1 and y1 - y0 == 1:
            if self.world[x0:x1, y0:y1] == obs:
                sw.append((x0, y0, x1, y1))
            return

        w, h = int(x1 - x0), int(y1 - y0)
        w2, h2 = int((x1 - x0) / 2), int((y1 - y0) / 2)

        # the four quadrants of subworld
        qA = 0 + x0, 0 + y0, w2 + x0, h2 + y0  # top left
        qB = w2 + x0, 0 + y0, w + x0, h2 + y0  # top right
        qC = 0 + x0, h2 + y0, w2 + x0, h + y0  # bottom left
        qD = w2 + x0, h2 + y0, w + x0, h + y0  # bottom right

        for (qx0, qy0, qx1, qy1) in (qA, qB, qC, qD):
            subworld = self.world[qx0:qx1, qy0:qy1]
            if obs:
                if np.all(subworld):
                    homog = True
                else:
                    homog = False
            else:
                if np.all(~subworld):
                    homog = True
                else:
                    homog = False

            if not homog:
                self.subdivide(qx0, qy0, qx1, qy1, sw=sw, obs=obs)
            else:
                sw.append((qx0, qy0, qx1, qy1))
                continue

    def get_rect_patches_mpl(self, c="k", lw=1, fc="none"):
        rects = []
        for (x0, y0, x1, y1) in self.obstacles:
            w = x1 - x0
            h = y1 - y0
            xy = (x0, y0)
            rects.append(Rectangle(xy, w, h, linewidth=lw, edgecolor=c, facecolor=fc))
        return rects

    def plot_rects(self, ax, c="k", lw=1, fc="none"):
        for r in self.get_rect_patches_mpl(c=c, lw=lw, fc=fc):
            ax.add_patch(r)
        ax.autoscale_view()
        return ax

    def get_obstacles(self):
        return self.obstacles

    def get_rand_start_end(self, bias=True):
        """if bias, prefer points far away from one another"""
        free_space = np.argwhere(self.world == 0)
        if bias == True:
            start_i = int(np.random.beta(a=0.5, b=5) * free_space.shape[0])
            end_i = int(np.random.beta(a=5, b=0.5) * free_space.shape[0])
        else:
            start_i = np.random.choice(free_space.shape[0])
            end_i = np.random.choice(free_space.shape[0])
        start = free_space[start_i, :]
        end = free_space[end_i, :]
        return start, end


def make_world(
    shape: tuple,
    scale: tuple,
    dimensions: int = 2,
    thresh=0.3,
) -> np.ndarray:

    if dimensions == 3:
        t, w, h = shape
        ts, ws, hs = scale

        noise1 = perlin_numpy.generate_perlin_noise_3d(
            (t, w, h),
            (ts, ws, hs),
            tileable=(True, False, False),
        )
        noise2 = perlin_numpy.generate_perlin_noise_3d(
            (t, w, h),
            (ts, ws * 2, hs * 2),
            tileable=(True, False, False),
        )

        noise = (noise1 + noise2) / 2
        noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
        return np.where(noise < thresh, 1, 0)

    elif dimensions == 2:
        w, h = shape
        ws, hs = scale

        noise1 = perlin_numpy.generate_perlin_noise_2d(
            (w, h),
            (ws, hs),
            tileable=(False, False),
        )
        noise2 = perlin_numpy.generate_perlin_noise_2d(
            (w, h),
            (ws * 2, hs * 2),
            tileable=(False, False),
        )

        noise = (noise1 + noise2) / 2
        noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
        return np.where(noise < thresh, 1, 0)


def get_rand_start_end(world: np.ndarray, bias=True) -> Tuple[np.ndarray, np.ndarray]:
    """if bias, prefer points far away from one another"""
    free_space = np.argwhere(world == 0)
    if bias == True:
        start_i = int(np.random.beta(a=0.5, b=5) * free_space.shape[0])
        end_i = int(np.random.beta(a=5, b=0.5) * free_space.shape[0])
    else:
        start_i = np.random.choice(free_space.shape[0])
        end_i = np.random.choice(free_space.shape[0])
    start = free_space[start_i, :]
    end = free_space[end_i, :]
    return start, end


def animate_perlin_world(world):
    fig = plt.figure()
    images = [
        [plt.imshow(layer, cmap="gray", interpolation="lanczos", animated=True)]
        for layer in world
    ]
    return animation.ArtistAnimation(fig, images, interval=50, blit=True)
