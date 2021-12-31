from world_gen import get_rand_start_end
import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from tqdm import tqdm
from scipy.ndimage import binary_dilation


DEFAULT_APPEARANCE = {
    "tree_color": "slategrey",
    "tree_alpha": 0.2,
    "world_cmap": "gist_gray_r",
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
        movespeed,
        r_within_goal,
        rrtobj,
        appearance=DEFAULT_APPEARANCE,
        buffer_size=None,
    ):
        self.movespeed = movespeed
        # radius within which we have "found" the goal
        self.r_within_goal = r_within_goal
        # RRT object. can be RRT star, informed, etc. Must have `make` method
        # which can take a start and goal and return a tree and a goal vector
        self.rrtobj = rrtobj
        # size of the "buffer" around obstacles. this is to avoid getting stuck
        # in the world. if the environment changes quickly, or if velocity is
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
        worlds,
    ):
        frames = worlds.shape[0]
        xstart, xgoal = get_rand_start_end(worlds[0], bias=True)
        current_position = xstart

        goals, positions = np.empty((frames, 2)), np.empty((frames, 2))
        paths, trees = [], []

        print("Simulating Dynamic Mission...")
        for fi in tqdm(range(frames), desc="frames"):
            world = worlds[fi]
            # if we are at the goal, get a new goal
            if np.linalg.norm(current_position - xgoal) < self.r_within_goal:
                _, xgoal = get_rand_start_end(world, bias=False)
            current_position = self.clamp(current_position, world.shape)

            # ok so we kinda want to avoid getting stuck in the world so
            # we'll do a binary dilation to get a bit of a buffer around objects
            # before computing the plan, but we remove only the buffer near us.
            bsize = int(self.movespeed / 2)
            world_dilated = binary_dilation(world, iterations=bsize)
            buffer_reg = world_dilated - world
            for i in range(bsize * 2):
                for j in range(bsize * 2):
                    hole = self.clamp(current_position + np.array([i, j]), world.shape)
                    # set as free
                    buffer_reg[hole[0], hole[1]] = 0
            world = world | buffer_reg
            if world[current_position[0], current_position[1]] == 1:
                in_obs = True
            else:
                in_obs = False
            self.rrtobj.set_world(world)
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

    def animate(self, goals, positions, paths, trees, worlds):
        fig, ax = plt.subplots()
        ax.set_xlim(0, worlds.shape[1])
        ax.set_ylim(0, worlds.shape[2])
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
            worlds[0].T,
            cmap=self.appearance["world_cmap"],
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
            im.set_data(worlds[i].T)
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

    def make_animation(self, worlds):
        results = self.simulate_dynamic_goals(worlds)
        return self.animate(*results, worlds)


if __name__ == "__main__":
    from rrt import RRTStarInformed
    from world_gen import make_world, get_rand_start_end

    w, h = 128, 128
    worlds = make_world((128, w, h), (16, 4, 4), dimensions=3, thresh=0.37)
    rrts = RRTStarInformed(worlds[0], n=800, r_rewire=128.0, r_goal=16.0, pbar=False)
    dyn_env_anim = DynamicEnvironmentAnimation(7.0, 5.0, rrts)
    anim = dyn_env_anim.make_animation(worlds)
    # anim.save("animation.mp4", fps=30, dpi=300)
    plt.show()
