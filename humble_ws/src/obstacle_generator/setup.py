# Copyright 2025 ReSim, Inc.
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
from glob import glob
from setuptools import find_packages, setup

package_name = "obstacle_generator"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Matt Coomber",
    maintainer_email="matt@resim.ai",
    description="A simple package for generating obstacles.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "obstacle_generator = obstacle_generator.obstacle_generator:main",
            "cleanup_obstacles = obstacle_generator.cleanup_obstacles:main",
        ],
    },
)
