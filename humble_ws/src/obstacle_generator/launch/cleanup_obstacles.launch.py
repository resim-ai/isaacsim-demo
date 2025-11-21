# Copyright 2025 ReSim, Inc.
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """
    Launch file to clean up all existing entities created by the obstacle_generator package.
    
    This launch file uses the Isaac Sim ROS 2 Simulation Control services to:
    1. Query for all entities matching the pattern "Obstacle[0-9]+" using GetEntities service
    2. Delete each found entity using DeleteEntity service
    
    Based on Isaac Sim documentation:
    https://docs.isaacsim.omniverse.nvidia.com/5.0.0/ros2_tutorials/tutorial_ros2_simulation_control.html#deleteentity-service
    """
    
    cleanup_node = Node(
        package="obstacle_generator",
        executable="cleanup_obstacles",
        name="obstacle_cleanup",
        output="screen",
    )
    
    return LaunchDescription([
        cleanup_node,
    ])

