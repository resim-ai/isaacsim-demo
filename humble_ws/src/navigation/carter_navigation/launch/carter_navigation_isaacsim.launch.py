## Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.

import json
import os
from pathlib import Path
from typing import Optional

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchDescriptionEntity
from launch.actions import TimerAction
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import RegisterEventHandler, ExecuteProcess, Shutdown
from launch.event_handlers import OnProcessIO
from launch.events.process import ProcessIO

def get_experience_location() -> str:
    test_config_path = Path("/tmp/resim/test_config.json")
    if test_config_path.exists():
        with open(test_config_path, "r") as f:
            test_config = json.load(f)
            return test_config["experienceLocation"]
    else:
        return os.path.join(get_package_share_directory("isaac_ros_navigation_goal"), "assets", "goals.txt")
    

def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time", default="True")

    map_dir = LaunchConfiguration(
        "map",
        default=os.path.join(
            get_package_share_directory("carter_navigation"),
            "maps",
            "carter_warehouse_navigation.yaml",
        ),
    )

    param_dir = LaunchConfiguration(
        "params_file",
        default=os.path.join(
            get_package_share_directory("carter_navigation"),
            "params",
            "carter_navigation_params.yaml",
        ),
    )

    nav2_bringup_launch_dir = os.path.join(
        get_package_share_directory("nav2_bringup"), "launch"
    )

    # rviz_config_dir = os.path.join(
    #     get_package_share_directory("carter_navigation"),
    #     "rviz2",
    #     "carter_navigation.rviz",
    # )

    topics = [
        # Core system topics
        "/clock",
        "/tf",
        
        # Navigation commands and status
        "/cmd_vel",
        "/cmd_vel_nav",
        "/initialpose",
        "/goal",
        "/goal/status",
        "/amcl_pose",
        "/behavior_tree_log",
        
        # Plans and trajectories
        "/plan",
        "/local_plan", 
        "/received_global_plan",
        "/transformed_global_plan",
        "/trajectories",
        
        # Costmaps
        "/global_costmap/costmap",
        "/global_costmap/costmap_raw",
        "/global_costmap/published_footprint",
        "/global_costmap/global_costmap/transition_event",
        "/local_costmap/costmap",
        "/local_costmap/costmap_raw",
        "/local_costmap/costmap_updates",
        "/local_costmap/published_footprint",
        "/local_costmap/clearing_endpoints",
        "/local_costmap/local_costmap/transition_event",
        
        # Sensor data
        "/chassis/odom",
        "/chassis/imu",
        "/front_stereo_imu/imu",
        "/left_stereo_imu/imu",
        "/front_stereo_camera/left/camera_info",
        "/front_stereo_camera/left/image_raw_throttled",
    ]

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
            "(/front_stereo_camera/left/image_raw$|/front_stereo_camera/left/image_raw/nitros_bridge$)"
        ],
        # ] + topics,
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
            "goal_text_file_path": get_experience_location()
        }.items(),
    )

    ld_record_and_start = LaunchDescription(
        [record_node, TimerAction(period=5.0, actions=[ld_automatic_goal])]
    )

    def execute_second_node_if_condition_met(event: ProcessIO, second_node_action: LaunchDescriptionEntity, message: str) -> Optional[LaunchDescriptionEntity]:
        output = event.text.decode().strip()
        # Look for fully loaded message from Isaac Sim. Only applicable in Gui mode.
        if message in output:
            # Log a message indicating the condition has been met
            print("Condition met, launching the node.")

            return second_node_action

    return LaunchDescription(
        [
            # # Declaring the Isaac Sim scene path. 'gui' launch argument is already used withing run_isaac_sim.launch.py
            # DeclareLaunchArgument(
            #     "gui",
            #     default_value="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Samples/ROS2/Scenario/carter_warehouse_navigation.usd",
            #     description="Path to isaac sim scene",
            # ),
            # # Include Isaac Sim launch file from isaacsim package with given launch parameters.
            # IncludeLaunchDescription(
            #     PythonLaunchDescriptionSource(
            #         [
            #             os.path.join(
            #                 get_package_share_directory("isaacsim"),
            #                 "launch",
            #                 "run_isaacsim.launch.py",
            #             ),
            #         ]
            #     ),
            #     launch_arguments={
            #         "version": "4.5.0",
            #         "play_sim_on_start": "true",
            #         "install_path": "/isaac-sim",
            #         "headless": "webrtc",
            #     }.items(),
            # ),
            # DeclareLaunchArgument("map", default_value=map_dir, description="Full path to map file to load"),
            # DeclareLaunchArgument(
            #     "params_file", default_value=param_dir, description="Full path to param file to load"
            # ),
            # DeclareLaunchArgument(
            #     "use_sim_time", default_value="true", description="Use simulation (Omniverse Isaac Sim) clock if true"
            # ),
            # IncludeLaunchDescription(
            #     PythonLaunchDescriptionSource(os.path.join(nav2_bringup_launch_dir, "rviz_launch.py")),
            #     launch_arguments={"namespace": "", "use_namespace": "False", "rviz_config": rviz_config_dir}.items(),
            # ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [nav2_bringup_launch_dir, "/bringup_launch.py"]
                ),
                launch_arguments={
                    "map": map_dir,
                    "use_sim_time": use_sim_time,
                    "params_file": param_dir,
                }.items(),
            ),
            Node(
                name="image_throttler",
                package="topic_tools",
                executable="throttle",
                output="screen",
                arguments=["messages", "/front_stereo_camera/left/image_raw", "5.0", "/front_stereo_camera/left/image_raw_throttled"]
            ),
            Node(
                package="pointcloud_to_laserscan",
                executable="pointcloud_to_laserscan_node",
                remappings=[
                    ("cloud_in", ["/front_3d_lidar/lidar_points"]),
                    ("scan", ["/scan"]),
                ],
                parameters=[
                    {
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
                        # 'concurrency_level': 1,
                    }
                ],
                name="pointcloud_to_laserscan",
            ),
            # Launch recorder and automatic goal generator node when Isaac Sim has finished loading.
            RegisterEventHandler(
                OnProcessIO(
                    on_stderr=lambda event: execute_second_node_if_condition_met(
                        event,
                        ld_record_and_start,
                        "[global_costmap.global_costmap]: start",
                    )
                )
            ),
            # Shut down when all goals reached
            RegisterEventHandler(
                OnProcessIO(
                    on_stderr=lambda event: execute_second_node_if_condition_met(
                        event,
                        TimerAction(period=5.0, actions=[Shutdown(reason="All goals completed.")]),
                        "All goals reached.",
                    )
                )
            )
        ]
    )
