# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from ament_index_python.packages import get_package_share_directory
from isaac_ros_navigation_goal.experience_config import load_experience_config
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.events.process import ProcessExited
from launch.launch_context import LaunchContext

def exit_handler(event: ProcessExited, context: LaunchContext):
    if event.returncode == 0:
        return [Shutdown(reason="All goals completed.")]
    return [Shutdown(reason="Navigation goal process failed.")]

def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time", default=True)
    experience_config = load_experience_config(
        default_initial_pose=[-6.4, -1.04, 0.0, 0.02, 0.0, 0.0, 0.99]
    )

    namespace = LaunchConfiguration("namespace", default="carter1")

    map_yaml_file = LaunchConfiguration(
        "map_yaml_path",
        default=os.path.join(
            get_package_share_directory("isaac_ros_navigation_goal"), "assets", "carter_warehouse_navigation.yaml"
        ),
    )

    goal_text_file = LaunchConfiguration(
        "goal_text_file_path",
        default=experience_config["goals_path"],
    )

    initial_pose = LaunchConfiguration(
        "initial_pose",
        default=str(experience_config["initial_pose"]),
    )
    publish_initial_pose = LaunchConfiguration("publish_initial_pose", default="true")

    navigation_goal_node = Node(
        name="set_navigation_goal",
        package="isaac_ros_navigation_goal",
        executable="SetNavigationGoal",
        namespace=namespace,
        parameters=[
            {
                "map_yaml_path": map_yaml_file,
                "iteration_count": experience_config["goal_count"],
                "goal_generator_type": "GoalReader",
                "action_server_name": "navigate_to_pose",
                "obstacle_search_distance_in_meters": 0.2,
                "goal_text_file_path": goal_text_file,
                "initial_pose": initial_pose,
                "publish_initial_pose": publish_initial_pose,
                "use_sim_time": use_sim_time,
                # Production: increase initial_pose_settle_sec (e.g. 3.0) if bt_navigator reports "Failed to send goal response"
                "initial_pose_settle_sec": 0.5,
                "max_goal_send_retries": 3,
                "server_wait_timeout_sec": 60.0,
            }
        ],
        output="screen",
    )

    return LaunchDescription([
        navigation_goal_node,
        RegisterEventHandler(
            OnProcessExit(
                target_action=navigation_goal_node,
                on_exit=exit_handler
            )
        )
    ])
