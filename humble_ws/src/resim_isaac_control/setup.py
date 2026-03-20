# Copyright 2025 ReSim, Inc.
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from setuptools import find_packages, setup

package_name = "resim_isaac_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Matt Coomber",
    maintainer_email="matt@resim.ai",
    description="Isaac Sim readiness and simulator-control nodes for ReSim workflows.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "isaac_ready = resim_isaac_control.isaac_ready:main",
            "set_simulation_state = resim_isaac_control.set_simulation_state:main",
        ],
    },
)
