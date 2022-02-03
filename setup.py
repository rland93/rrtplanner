from setuptools import find_packages, setup
import pathlib

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="rrtplanner",
    version="1.0.0",
    description="A (partially) vectorized RRT, RRT*, RRT*Informed planner for 2-D occupancyGrids.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/rland93/rrt_pathplanner",
    author="rland93",
    author_email="msutherl@uci.edu",
    license="MIT",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "cvxpy>=1.1",
        "matplotlib>=3.5",
        "networkx>=2.6",
        "numba==0.55",
        "numpy>=1.20",
        "pyfastnoisesimd==0.4",
        "pytest==6.2",
        "scipy==1.7",
        "tqdm==4.62",
    ],
)
