# Copyright 2025 ReSim, Inc.
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import json
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def get_experience_location() -> str:
    """Get the goals file path from test config or use default."""
    test_config_path = Path("/tmp/resim/test_config.json")
    if test_config_path.exists():
        with open(test_config_path, "r") as f:
            test_config = json.load(f)
            return test_config["experienceLocation"]
    else:
        return os.path.join(
            get_package_share_directory("isaac_ros_navigation_goal"), "assets", "goals.txt"
        )


def generate_launch_description():
    # Default map path
    default_map_path = os.path.join(
        get_package_share_directory("carter_navigation"),
        "maps",
        "carter_warehouse_navigation.yaml",
    )

    # Launch arguments
    map_yaml_path_arg = DeclareLaunchArgument(
        "map_yaml_path",
        default_value=default_map_path,
        description="Path to the map YAML file",
    )

    goal_text_file_path_arg = DeclareLaunchArgument(
        "goal_text_file_path",
        default_value=get_experience_location(),
        description="Path to the goals text file",
    )

    seed_arg = DeclareLaunchArgument(
        "seed",
        default_value="-1",
        description="Random seed for reproducible pallet generation. Leave unset or use -1 for random generation.",
    )

    num_obstacles_arg = DeclareLaunchArgument(
        "num_obstacles",
        default_value="20",
        description="Number of pallets to spawn.",
    )

    initial_pose_arg = DeclareLaunchArgument(
        "initial_pose",
        default_value="[-6.4, -1.04, 0.0, 0.0, 0.0, 0.99, 0.02]",
        description="Initial pose of the robot [x, y, z, qx, qy, qz, qw] to avoid spawning obstacles near it.",
    )

    # Node
    obstacle_generator_node = Node(
        package="obstacle_generator",
        executable="obstacle_generator",
        name="obstacle_generator",
        parameters=[
            {
                "map_yaml_path": LaunchConfiguration("map_yaml_path"),
                "goal_text_file_path": LaunchConfiguration("goal_text_file_path"),
                "seed": LaunchConfiguration("seed"),
                "num_obstacles": LaunchConfiguration("num_obstacles"),
                "initial_pose": LaunchConfiguration("initial_pose"),
            }
        ],
        output="screen",
    )

    return LaunchDescription([
        map_yaml_path_arg,
        goal_text_file_path_arg,
        seed_arg,
        num_obstacles_arg,
        initial_pose_arg,
        obstacle_generator_node,
    ])

