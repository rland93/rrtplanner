from setuptools import find_packages, setup

setup(
    name="rrtpp",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3",
    url="https://github.com/rland93/rrt_pathplanner",
    author="rland93",
    license="MIT",
    install_requires=[
        "numpy>=1.15",
        "matplotlib>=3.4.3",
        "networkx>=2.6.3",
        "uuid>=1.30",
        "scipy>=1.7.1",
    ],
    zip_safe=False,
)
