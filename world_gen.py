from matplotlib.collections import PatchCollection
import numpy as np
from matplotlib.patches import Rectangle
from perlin_noise.perlin_noise import PerlinNoise
import rtree, uuid


class ObstacleGenerator(object):
    def __init__(self, superworld):
        self.superworld = superworld
        self.obstacles = self.generate_subdivisions(obs=True)
        self.free_space = self.generate_subdivisions(obs=False)
        self.free_space_list = np.argwhere(self.superworld == 0)
        self.obs_rtree = self.get_rtree(self.obstacles)
        self.fre_rtree = self.get_rtree(self.free_space)

    def generate_subdivisions(self, obs=True):
        ax0 = 0
        ay0 = 0
        ax1 = self.superworld.shape[0]
        ay1 = self.superworld.shape[1]
        sw = []
        self.subdivide(ax0, ay0, ax1, ay1, sw=sw, obs=obs)
        return sw

    def subdivide(self, x0, y0, x1, y1, sw=[], obs=True):
        """Recursive quadtree subdivision algorithm"""
        if x1 - x0 == 1 and y1 - y0 == 1:
            if self.superworld[x0:x1, y0:y1] == obs:
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
            subworld = self.superworld[qx0:qx1, qy0:qy1]
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
        free_space = np.argwhere(self.superworld == 0)
        if bias == True:
            start_i = int(np.random.beta(a=0.5, b=5) * free_space.shape[0])
            end_i = int(np.random.beta(a=5, b=0.5) * free_space.shape[0])
        else:
            start_i = np.random.choice(free_space.shape[0])
            end_i = np.random.choice(free_space.shape[0])
        start = free_space[start_i, :] + np.array((0.5, 0.5))
        end = free_space[end_i, :] + np.array((0.5, 0.5))
        return start, end

    def get_rtree(self, obstacles=None):
        stream = [(uuid.uuid4(), o, None) for o in obstacles]
        p = rtree.index.Property(dimension=2)
        return rtree.index.Index(stream, properties=p, interleaved=True)


def make_perlin_world(w, h, octaves=5, seed=None, thresh=0.3):
    # create noise function
    noise = PerlinNoise(octaves=5, seed=seed)
    # create perlin world
    world = np.array([[noise([i / w, j / h]) for j in range(h)] for i in range(w)])
    # scale world to [0, 1]
    world = (world - np.min(world)) / (np.max(world) - np.min(world))
    # binary threshold on world
    return np.where(world > thresh, 0, 1)
