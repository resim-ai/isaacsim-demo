# Modified from the original carter_navigation_isaacsim.launch.py file by NVIDIA Corporation.
# Source: https://github.com/isaac-sim/IsaacSim-ros_workspaces/blob/1e70d8169bc049498bca87f1459b1e1f4f133447/humble_ws/src/navigation/carter_navigation/launch/carter_navigation_isaacsim.launch.py
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
from pathlib import Path
from typing import Optional

from ament_index_python.packages import get_package_share_directory
from isaac_ros_navigation_goal.experience_config import TEST_CONFIG_PATH, load_experience_config  # pyright: ignore[reportMissingImports]
from launch import LaunchDescription, LaunchDescriptionEntity
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    Shutdown,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit, OnProcessIO
from launch.events.process import ProcessIO
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def execute_action_if_message_seen(
    event: ProcessIO, action: LaunchDescriptionEntity, message: str
) -> Optional[LaunchDescriptionEntity]:
    output = event.text.decode().strip()
    if message in output:
        print("Condition met, launching the node.")
        return action


def require_experience_field(experience_config: dict, key: str, *, allow_empty: bool = False) -> str:
    value = experience_config.get(key)
    if value is None or (not allow_empty and value == ""):
        raise ValueError(f"Experience file must define '{key}' for the hospital navigation launch.")
    return value


def namespaced_topic(namespace: str, topic: str) -> str:
    topic = topic.lstrip("/")
    if namespace:
        return f"/{namespace}/{topic}"
    return f"/{topic}"


def resolve_requested_experience_path(context) -> Optional[Path]:
    explicit_experience_path = LaunchConfiguration("experience_path").perform(context).strip()
    if TEST_CONFIG_PATH.exists() or not explicit_experience_path:
        return None
    return Path(explicit_experience_path)


def launch_setup(context):
    # -------------------------------------------------------------------------
    # Runtime configuration
    # -------------------------------------------------------------------------
    use_sim_time = LaunchConfiguration("use_sim_time", default="True")
    requested_experience_path = resolve_requested_experience_path(context)
    experience_config = load_experience_config(experience_location=requested_experience_path)
    namespace = require_experience_field(experience_config, "namespace", allow_empty=True)
    use_namespace = "True" if namespace else "False"

    # -------------------------------------------------------------------------
    # Paths and launch argument defaults
    # -------------------------------------------------------------------------
    carter_navigation_share_dir = get_package_share_directory("carter_navigation")
    nav2_bringup_launch_dir = os.path.join(
        get_package_share_directory("nav2_bringup"), "launch"
    )
    rviz_config_dir = os.path.join(
        carter_navigation_share_dir,
        "rviz2",
        require_experience_field(experience_config, "rviz_config_file"),
    )

    map_dir = LaunchConfiguration(
        "map",
        default=os.path.join(
            carter_navigation_share_dir,
            "maps",
            require_experience_field(experience_config, "map_file"),
        ),
    )
    param_dir = LaunchConfiguration(
        "params_file",
        default=os.path.join(
            carter_navigation_share_dir,
            "params",
            require_experience_field(experience_config, "nav2_params_file"),
        ),
    )

    # -------------------------------------------------------------------------
    # Core nodes
    # -------------------------------------------------------------------------
    isaac_ready_node = Node(
        package="resim_isaac_control",
        executable="isaac_ready",
        name="isaac_ready_node",
        namespace=namespace,
        parameters=[
            {
                "initial_pose": experience_config["initial_pose"],
                "isaacsim_entity": experience_config["isaacsim_entity"],
                "world_uri": experience_config["world_uri"],
            }
        ],
    )
    publish_initial_pose_node = Node(
        package="carter_navigation",
        executable="publish_initial_pose",
        name="publish_initial_pose",
        namespace=namespace,
        output="screen",
        parameters=[
            {
                "initial_pose": experience_config["initial_pose"],
                "frame_id": "map",
                "wait_for_subscribers_timeout_sec": 120.0,
                "publish_repetitions": 3,
                "publish_period_sec": 1.0,
            }
        ],
    )
    set_simulation_playing_node = Node(
        package="resim_isaac_control",
        executable="set_simulation_state",
        name="set_simulation_playing",
        namespace=namespace,
        output="screen",
        parameters=[{"target_state": "playing"}],
    )
    set_simulation_stopped_node = Node(
        package="resim_isaac_control",
        executable="set_simulation_state",
        name="set_simulation_stopped",
        namespace=namespace,
        output="screen",
        parameters=[{"target_state": "stopped"}],
    )
    metrics_emitter_node = Node(
        package="metrics_emitter",
        executable="metrics_emitter",
        output="screen",
        namespace=namespace,
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "goal_count": experience_config["goal_count"],
            }
        ],
        remappings=(
            [
                ("/tf", f"/{namespace}/tf"),
                ("/tf_static", f"/{namespace}/tf_static"),
            ]
            if namespace
            else []
        ),
    )
    image_throttler_node = Node(
        name="image_throttler",
        package="topic_tools",
        executable="throttle",
        output="screen",
        namespace=namespace,
        arguments=[
            "messages",
            namespaced_topic(namespace, "front_stereo_camera/left/image_raw"),
            "5.0",
            namespaced_topic(namespace, "front_stereo_camera/left/image_raw_throttled"),
        ],
    )
    pointcloud_to_laserscan_node = Node(
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        name="pointcloud_to_laserscan",
        namespace=namespace,
        remappings=[
            ("cloud_in", namespaced_topic(namespace, "front_3d_lidar/lidar_points")),
            ("scan", namespaced_topic(namespace, "scan")),
        ],
        parameters=[
            {
                "use_sim_time": True,
                "target_frame": "front_3d_lidar",
                "transform_tolerance": 0.01,
                "min_height": -0.4,
                "max_height": 1.5,
                "angle_min": -1.5708,  # -M_PI/2
                "angle_max": 1.5708,  # M_PI/2
                "angle_increment": 0.0087,  # M_PI/360.0
                "scan_time": 0.3333,
                "range_min": 0.05,
                "range_max": 100.0,
                "use_inf": True,
                "inf_epsilon": 1.0,
                # "concurrency_level": 1,
            }
        ],
    )

    # -------------------------------------------------------------------------
    # Deferred launch actions triggered by process output
    # -------------------------------------------------------------------------
    record_node = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "record",
            "--storage",
            "mcap",
            "--output",
            "/tmp/resim/outputs/record",
            "--use-sim-time",
            "--all",
            "--exclude",
            "(/front_stereo_camera/left/image_raw$|/front_stereo_camera/left/image_raw/nitros_bridge$)",
        ],
        output="screen",
    )
    ld_automatic_goal = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("isaac_ros_navigation_goal"),
                    "launch",
                    "isaac_ros_navigation_goal.launch.py",
                ),
            ]
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "experience_path": experience_config["experience_path"],
            "initial_pose": str(experience_config["initial_pose"]),
            "goal_count": str(experience_config["goal_count"]),
            "publish_initial_pose": "false",
            "shutdown_on_exit": "false",
            "namespace": namespace,
        }.items(),
    )
    ld_record_and_start = LaunchDescription(
        [record_node, TimerAction(period=1.0, actions=[ld_automatic_goal])]
    )
    ld_stop_simulation = LaunchDescription([set_simulation_stopped_node])

    # -------------------------------------------------------------------------
    # Main nav2 stack (started after isaac_ready passes)
    # -------------------------------------------------------------------------
    nav2_actions = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(nav2_bringup_launch_dir, "rviz_launch.py")),
            launch_arguments={
                "namespace": namespace,
                "use_namespace": use_namespace,
                "rviz_config": rviz_config_dir,
            }.items(),
            condition=IfCondition(LaunchConfiguration("rviz")),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([nav2_bringup_launch_dir, "/bringup_launch.py"]),
            launch_arguments={
                "namespace": namespace,
                "use_namespace": use_namespace,
                "map": map_dir,
                "use_sim_time": use_sim_time,
                "params_file": param_dir,
            }.items(),
        ),
        metrics_emitter_node,
        image_throttler_node,
        pointcloud_to_laserscan_node,
    ]

    nav2_event_handlers = [
        # Launch recorder and automatic goal generator node when Isaac Sim has finished loading.
        RegisterEventHandler(
            OnProcessIO(
                on_stderr=lambda event: execute_action_if_message_seen(
                    event,
                    ld_record_and_start,
                    "global_costmap.global_costmap]: start",
                )
            ),
            condition=IfCondition(LaunchConfiguration("send_goals")),
        ),
        RegisterEventHandler(
            OnProcessIO(
                on_stderr=lambda event: execute_action_if_message_seen(
                    event,
                    publish_initial_pose_node,
                    "amcl]: Activating",
                )
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=publish_initial_pose_node,
                on_exit=lambda event, context: [set_simulation_playing_node]
                if event.returncode == 0
                else [Shutdown(reason="PublishInitialPose failed; not starting simulation playback.")],
            )
        ),
        # Shut down when all goals reached
        RegisterEventHandler(
            OnProcessIO(
                on_stderr=lambda event: execute_action_if_message_seen(
                    event,
                    ld_stop_simulation,
                    "All goals reached.",
                )
            ),
            condition=IfCondition(LaunchConfiguration("send_goals")),
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=set_simulation_stopped_node,
                on_exit=lambda event, context: [
                    Shutdown(reason="All goals completed and simulation stopped.")
                ]
                if event.returncode == 0
                else [Shutdown(reason="Failed to stop simulation after goals completed.")],
            ),
            condition=IfCondition(LaunchConfiguration("send_goals")),
        ),
    ]

    nav2_stack = nav2_actions + nav2_event_handlers

    nav2_stack_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=isaac_ready_node,
            on_exit=lambda event, context: nav2_stack
            if event.returncode == 0
            else [Shutdown(reason="Isaac readiness checks failed; not starting Nav2 stack.")],
        )
    )

    # -------------------------------------------------------------------------
    # Top-level launch description
    # -------------------------------------------------------------------------
    return [
        isaac_ready_node,
        nav2_stack_handler,
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "rviz", default_value="false", description="Launch RViz if true"
            ),
            DeclareLaunchArgument(
                "send_goals", default_value="true", description="Send goals if true"
            ),
            DeclareLaunchArgument(
                "experience_path",
                default_value="",
                description=(
                    "Path to an experience YAML file to use when "
                    "/tmp/resim/test_config.json is not present"
                ),
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )

