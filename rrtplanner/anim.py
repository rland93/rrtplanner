import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from tqdm import tqdm
from scipy.ndimage import binary_dilation
from rrt import random_point_og, RRT


DEFAULT_APPEARANCE = {
    "tree_color": "slategrey",
    "tree_alpha": 0.2,
    "og_cmap": "gist_gray_r",
    "goal_marker": "*",
    "goal_color": "seagreen",
    "goal_circ_color": "seagreen",
    "start_marker": "o",
    "start_color": "red",
    "path_color": "blue",
}


class DynamicEnvironmentAnimation(object):
    def __init__(
        self,
        movespeed: float,
        r_within_goal: float,
        rrtobj: RRT,
        appearance: dict = DEFAULT_APPEARANCE,
        buffer_size: int = None,
    ):
        self.movespeed = movespeed
        # radius within which we have "found" the goal
        self.r_within_goal = r_within_goal
        # RRT object. can be RRT star, informed, etc. Must have `make` method
        # which can take a start and goal and return a tree and a goal vector
        self.rrtobj = rrtobj
        # size of the "buffer" around obstacles. this is to avoid getting stuck
        # in the obstacles. if the environment changes quickly, or if velocity is
        # large, this should be a larger value. But setting a very large value
        # will prevent optimal paths from being found.
        if buffer_size is not None:
            self.buffer_size = buffer_size
        else:
            self.buffer_size = int(movespeed / 2)
        self.appearance = appearance

    @staticmethod
    def clamp(xy, shape):
        x = max(0, min(xy[0], shape[0] - 1))
        y = max(0, min(xy[1], shape[1] - 1))
        return np.array([x, y])

    def simulate_dynamic_goals(
        self,
        og_3d,
    ):
        frames = og_3d.shape[0]
        xstart = random_point_og(og_3d[0])
        xgoal = random_point_og(og_3d[0])
        current_position = xstart

        goals, positions = np.empty((frames, 2)), np.empty((frames, 2))
        paths, trees = [], []

        print("Simulating Dynamic Mission...")
        for fi in tqdm(range(frames), desc="frames"):
            og = og_3d[fi]
            # if we are at the goal, get a new goal
            if np.linalg.norm(current_position - xgoal) < self.r_within_goal:
                xgoal = random_point_og(og)
            current_position = self.clamp(current_position, og.shape)

            # ok so we kinda want to avoid getting stuck in the obstacles so
            # we'll do a binary dilation to get a bit of a buffer around objects
            # before computing the plan, but we remove only the buffer near us.
            bsize = int(self.movespeed / 2)
            og_dilated = binary_dilation(og, iterations=bsize)
            buffer_reg = og_dilated - og
            for i in range(bsize * 2):
                for j in range(bsize * 2):
                    hole = self.clamp(current_position + np.array([i, j]), og.shape)
                    # set as free
                    buffer_reg[hole[0], hole[1]] = 0
            og = og | buffer_reg
            if og[current_position[0], current_position[1]] == 1:
                in_obs = True
            else:
                in_obs = False
            self.rrtobj.set_og(og)
            T, gv = self.rrtobj.make(current_position, xgoal)
            path = self.rrtobj.path_points(T, self.rrtobj.route2gv(T, gv))

            # populate lists
            goals[fi] = xgoal
            positions[fi] = current_position
            paths.append(path)
            tree = []
            for e1, e2 in T.edges:
                p1, p2 = T.nodes[e1]["pt"], T.nodes[e2]["pt"]
                tree.append([p1, p2])
            trees.append(np.array(tree))

            if path.shape[0] != 0 and not in_obs:
                # follow current plan
                v = path[0, 1, :] - path[0, 0, :]
                angle = np.arctan2(v[1], v[0])
                # update simulation
                current_position += np.int64(
                    np.array([np.cos(angle), np.sin(angle)]) * self.movespeed
                )

        return goals, positions, paths, trees

    def animate(self, goals, positions, paths, trees, og_3d):
        fig, ax = plt.subplots()
        ax.set_xlim(0, og_3d.shape[1])
        ax.set_ylim(0, og_3d.shape[2])
        ax.set_aspect("equal")

        pathlc = LineCollection(
            [], color=self.appearance["path_color"], zorder=3, linewidth=2
        )
        treelc = LineCollection(
            [],
            color=self.appearance["tree_color"],
            zorder=2,
            alpha=self.appearance["tree_alpha"],
        )
        im = ax.imshow(
            og_3d[0].T,
            cmap=self.appearance["og_cmap"],
            vmin=0,
            vmax=1,
            interpolation=None,
        )

        goal_sc = ax.scatter(
            [],
            [],
            marker=self.appearance["goal_marker"],
            color=self.appearance["goal_color"],
            s=40,
            zorder=4,
        )
        goalcircle = Circle(
            [0, 0],
            radius=self.r_within_goal,
            fill=None,
            ec=self.appearance["goal_circ_color"],
            zorder=3,
        )
        start_sc = ax.scatter(
            [],
            [],
            marker=self.appearance["start_marker"],
            color=self.appearance["start_color"],
            s=40,
            zorder=4,
        )

        def init():
            ax.add_collection(pathlc)
            ax.add_collection(treelc)
            ax.add_artist(goalcircle)
            return im, pathlc, treelc, goal_sc, goalcircle, start_sc

        def update(i):
            pathlc.set_segments(paths[i])
            treelc.set_segments(trees[i])
            im.set_data(og_3d[i].T)
            goal_sc.set_offsets([goals[i, 0], goals[i, 1]])
            goalcircle.center = goals[i]
            start_sc.set_offsets([positions[i, 0], positions[i, 1]])
            return im, pathlc, treelc, goal_sc, goalcircle, start_sc

        anim = FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=goals.shape[0],
            interval=50,
            blit=True,
        )
        return anim

    def make_animation(self, og_3d):
        results = self.simulate_dynamic_goals(og_3d)
        return self.animate(*results, og_3d)


if __name__ == "__main__":
    import rrt
    from oggen import perlin_occupancygrid

    w, h = 100, 100
    frames = 200
    og_3d = perlin_occupancygrid(w, h, 0.33, frames)
    rrts = rrt.RRTStarInformed(og_3d[0], n=1000, r_rewire=25, r_goal=12.5, pbar=False)

    dyn_env_anim = DynamicEnvironmentAnimation(7.0, 5.0, rrts)
    anim = dyn_env_anim.make_animation(og_3d)
    # anim.save("animation.mp4", fps=30, dpi=300)
    plt.show()
