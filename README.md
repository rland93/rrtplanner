# RRT Path planning with Python!

![Screenshot of RRTstar path](https://github.com/rland93/rrtplanner/blob/main/figure.png)

This repository contains my planning algorithm research code. It has modules for creating random or dynamic occupancy grids on which to generate plans, utilities for plotting and animating plans, and implementations of the RRT, RRT*, and RRT-Informed planners. 

## Features

Utility Modules:
+ Random World Generation with Perlin Noise
+ Animation Module
+ Dubins Primitive Module

Planning Modules:
+ RRT Planner
+ RRT(star) Planner
+ RRT(informed) Planner
+ Dubins Vehicle RRT Planner
+ Dubins Vehicle RRT(star) Planner

Path Following Modules:
+ Straight Line Vector-Field Path-Follower
+ Decomposition of multi-waypoint plan for vector-field follower using voronoi regions

All algorithms are written in Python. I've tried to accelerate some things with Numba; so, your platform needs to be [compatible with numba](https://numba.readthedocs.io/en/stable/user/installing.html). But for the most part, it's standard python, and as such, is extremely slow.

This is research code, so it's not *at all* a robust package to use for a real planner. Let me know how you are using it!

## Dependency Notes

It relies on Numba to accelerate some collision logic:

https://numba.readthedocs.io/en/stable/user/installing.html

`pyfastnoisesimd` relies on SIMD instructions available only on modern processors to speed up perlin noise generation for random occupancyGrids. I'm not sure how it degrades if those instructions are not available. See https://pyfastnoisesimd.readthedocs.io/en/latest/overview.html.

## Information:

RRT:

+ [LaValle's RRT Page.](http://lavalle.pl/rrt/)
+ [RRT * Paper](https://arxiv.org/abs/1005.0416)
+ [Informed RRT* Paper](https://arxiv.org/abs/1404.2334v3)

Planners:

+ [Planning Algorithms](http://lavalle.pl/planning/)

Path Followers:

+ [Vector Field Path Following for Miniature Air Vehicles](https://ieeexplore.ieee.org/document/4252175)
+ [Unmanned Aerial Vehicle Path Following: A Survey and Analysis of Algorithms for Fixed-Wing Unmanned Aerial Vehicles](https://ieeexplore.ieee.org/document/6712082)
