import networkx as nx
import numpy as np
from rrtpp.rrt import RRTstar

if __name__ == "__main__":
    world_size = (512, 512)
    N = 5000
    r_rewire = 512
    r_goal = 40
    h, w = world_size
    world = np.zeros((1000, 1000), dtype=bool)
    world[250:750, 400:600] = 1
    world[450:480, 400:600] = 0
    xstart, xgoal = np.array([500, 50]), np.array([500, 950])
    # RRT Star
    rrt_star = RRTstar(world, N)
    T = rrt_star.make(xstart, xgoal, 50)
