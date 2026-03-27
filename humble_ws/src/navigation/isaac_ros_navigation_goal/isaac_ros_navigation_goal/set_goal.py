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

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from action_msgs.msg import GoalStatus as ActionGoalStatus
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Header
from .obstacle_map import GridMap
from .goal_generators import RandomGoalGenerator, GoalReader
import sys
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from custom_message.msg import GoalStatus
import threading


class SetNavigationGoal(Node):
    def __init__(self):
        super().__init__("set_navigation_goal")

        self.declare_parameter("iteration_count", 1)
        self.declare_parameter("goal_generator_type", "RandomGoalGenerator")
        self.declare_parameter("action_server_name", "navigate_to_pose")
        self.declare_parameter("obstacle_search_distance_in_meters", 0.2)
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("map_yaml_path", "")
        self.declare_parameter("experience_path", "")
        self.declare_parameter("initial_pose", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.declare_parameter("publish_initial_pose", True)
        self.declare_parameter("max_goal_send_retries", 3)
        self.declare_parameter("initial_pose_settle_sec", 0.5)
        self.declare_parameter("server_wait_timeout_sec", 60.0)

        self.__goal_generator = self.__create_goal_generator()
        action_server_name = self.get_parameter("action_server_name").value
        namespace = self.get_namespace()
        if namespace == "/":
            namespace = ""
        self._action_client = ActionClient(self, NavigateToPose, f"{namespace}/{action_server_name}")

        self.MAX_ITERATION_COUNT = int(self.get_parameter("iteration_count").value)
        assert self.MAX_ITERATION_COUNT > 0
        self.curr_iteration_count = 1

        goal_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.__initial_goal_publisher = self.create_publisher(PoseWithCovarianceStamped, "initialpose", 5)
        self.__goal_publisher = self.create_publisher(PoseStamped, "goal", goal_qos)
        self.__goal_status_publisher = self.create_publisher(GoalStatus, "goal/status", goal_qos)

        self.__initial_pose = self.get_parameter("initial_pose").value
        self.__publish_initial_pose = bool(self.get_parameter("publish_initial_pose").value)
        self.__is_initial_pose_sent = (not self.__publish_initial_pose) or (self.__initial_pose is None)
        self._goal_send_retries = 0
        self._current_goal_msg = None
        self._retry_timer = None

    def __send_initial_pose(self):
        """
        Publishes the initial pose.
        This function is only called once that too before sending any goal pose
        to the mission server.
        """
        goal = PoseWithCovarianceStamped()
        goal.header.frame_id = self.get_parameter("frame_id").value
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = self.__initial_pose[0]
        goal.pose.pose.position.y = self.__initial_pose[1]
        goal.pose.pose.position.z = self.__initial_pose[2]
        goal.pose.pose.orientation.w = self.__initial_pose[3]
        goal.pose.pose.orientation.x = self.__initial_pose[4]
        goal.pose.pose.orientation.y = self.__initial_pose[5]
        goal.pose.pose.orientation.z = self.__initial_pose[6]
        self.__initial_goal_publisher.publish(goal)

    def send_goal(self):
        """
        Sends the goal to the action server.
        """

        if not self.__is_initial_pose_sent:
            self.get_logger().info("Sending initial pose")
            self.__send_initial_pose()
            self.__is_initial_pose_sent = True

            # Give Nav2 time to be ready after initial pose (increase in production if goals are rejected).
            settle_sec = float(self.get_parameter("initial_pose_settle_sec").value)
            self.get_clock().sleep_for(Duration(seconds=settle_sec))
            self.get_logger().info("Sending first goal")

        timeout_sec = float(self.get_parameter("server_wait_timeout_sec").value)
        if not self._action_client.wait_for_server(timeout_sec=timeout_sec):
            self.get_logger().error("Action server not available after %.1f s" % timeout_sec)
            rclpy.shutdown()
            sys.exit(1)

        self.__goal_status_publisher.publish(GoalStatus(header=Header(stamp=self.get_clock().now().to_msg()), status="NEW_GOAL"))
        # Use cached goal only when retrying after rejection; otherwise get next goal.
        if self._goal_send_retries == 0:
            self._current_goal_msg = None
        goal_msg = self._current_goal_msg if self._current_goal_msg is not None else self.__get_goal()

        if goal_msg is None:
            rclpy.shutdown()
            sys.exit(1)

        self._current_goal_msg = goal_msg
        self.__goal_publisher.publish(goal_msg.pose)

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.__feedback_callback
        )
        self._send_goal_future.add_done_callback(self.__goal_response_callback)

    def __goal_response_callback(self, future):
        """
        Callback function to check the response(goal accepted/rejected) from the server.
        On rejection (e.g. RMW timeout "Failed to send goal response") retries up to
        max_goal_send_retries before exiting.
        """
        try:
            goal_handle = future.result()
        except Exception as e:
            self.get_logger().warn("Goal response failed: %s" % e)
            goal_handle = None

        if goal_handle is None or not goal_handle.accepted:
            self._goal_send_retries += 1
            max_retries = int(self.get_parameter("max_goal_send_retries").value)
            if self._goal_send_retries <= max_retries:
                self.get_logger().warn(
                    "Goal rejected or response lost (retry %d/%d). "
                    "Common in production when bt_navigator reports 'Failed to send goal response'. "
                    "Retrying in 2 s..."
                    % (self._goal_send_retries, max_retries)
                )
                if self._retry_timer is not None:
                    self._retry_timer.cancel()
                    self.destroy_timer(self._retry_timer)
                self._retry_timer = self.create_timer(2.0, self.__retry_send_goal)
                return
            self.get_logger().error("Goal rejected after %d retries." % max_retries)
            rclpy.shutdown()
            sys.exit(1)

        self._goal_send_retries = 0
        self.get_logger().info("Goal accepted :)")

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.__get_result_callback)

    def __retry_send_goal(self):
        """Resend only the action goal after a rejection/timeout (same goal, no new iteration). Topic /goal is not republished; it is assumed reliably delivered."""
        if self._retry_timer is not None:
            self._retry_timer.cancel()
            self.destroy_timer(self._retry_timer)
            self._retry_timer = None
        if self._current_goal_msg is None:
            return
        self._send_goal_future = self._action_client.send_goal_async(
            self._current_goal_msg, feedback_callback=self.__feedback_callback
        )
        self._send_goal_future.add_done_callback(self.__goal_response_callback)

    def __normalize_goal_pose(self, pose):
        """
        Normalize goal pose arrays to ROS pose fields.
        Preferred format is [x, y, z, qw, qx, qy, qz]. Legacy [x, y, qx, qy, qz, qw]
        remains supported for random goal generation.
        """
        if len(pose) == 7:
            return (pose[0], pose[1], pose[2], pose[4], pose[5], pose[6], pose[3])
        if len(pose) == 6:
            return (pose[0], pose[1], 0.0, pose[2], pose[3], pose[4], pose[5])
        self.get_logger().error(
            "Goal pose must contain either 7 values [x, y, z, qw, qx, qy, qz] "
            "or legacy 6 values [x, y, qx, qy, qz, qw]"
        )
        return None

    def __get_goal(self):
        """
        Get the next goal from the goal generator.

        Returns
        -------
        [NavigateToPose][goal] or None if the next goal couldn't be generated.

        """

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = self.get_parameter("frame_id").value
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        pose = self.__goal_generator.generate_goal()

        # couldn't sample a pose which is not close to obstacles. Rare but might happen in dense maps.
        if pose is None:
            self.get_logger().error(
                "Could not generate next goal. Returning. Possible reasons for this error could be:"
            )
            self.get_logger().error(
                "1. If you are using GoalReader then please make sure iteration count <= number of goals avaiable in file."
            )
            self.get_logger().error(
                "2. If RandomGoalGenerator is being used then it was not able to sample a pose which is given distance away from the obstacles."
            )
            return

        normalized_pose = self.__normalize_goal_pose(pose)
        if normalized_pose is None:
            return None

        self.get_logger().info("Generated goal pose: {0}".format(pose))
        goal_msg.pose.pose.position.x = normalized_pose[0]
        goal_msg.pose.pose.position.y = normalized_pose[1]
        goal_msg.pose.pose.position.z = normalized_pose[2]
        goal_msg.pose.pose.orientation.x = normalized_pose[3]
        goal_msg.pose.pose.orientation.y = normalized_pose[4]
        goal_msg.pose.pose.orientation.z = normalized_pose[5]
        goal_msg.pose.pose.orientation.w = normalized_pose[6]
        return goal_msg

    def __get_result_callback(self, future):
        """
        Callback to check result.\n
        It calls the send_goal() function in case current goal sent count < required goals count.     
        """
        goal_result = future.result()
        status = goal_result.status
        result = goal_result.result
        self.get_logger().info("Result status: %d, payload: %s" % (status, result.result))

        if status != ActionGoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error(
                "Goal did not succeed (status=%d). Not counting this as reached." % status
            )
            rclpy.shutdown()
            sys.exit(1)

        if self.curr_iteration_count < self.MAX_ITERATION_COUNT:
            self.curr_iteration_count += 1
            self.send_goal()
        else:
            self.__goal_status_publisher.publish(GoalStatus(header=Header(stamp=self.get_clock().now().to_msg()), status="COMPLETE"))
            self.get_logger().info("All goals reached.")
            rclpy.shutdown()

    def __feedback_callback(self, feedback_msg):
        """
        This is feeback callback. We can compare/compute/log while the robot is on its way to goal.
        """
        # self.get_logger().info('FEEDBACK: {}\n'.format(feedback_msg))
        pass

    def __create_goal_generator(self):
        """
        Creates the GoalGenerator object based on the specified ros param value.
        """

        goal_generator_type = self.get_parameter("goal_generator_type").value
        goal_generator = None
        if goal_generator_type == "RandomGoalGenerator":
            if self.get_parameter("map_yaml_path").value is None:
                self.get_logger().info("Yaml file path is not given. Returning..")
                sys.exit(1)

            yaml_file_path = self.get_parameter("map_yaml_path").value
            grid_map = GridMap(yaml_file_path)
            obstacle_search_distance_in_meters = self.get_parameter("obstacle_search_distance_in_meters").value
            assert obstacle_search_distance_in_meters > 0
            goal_generator = RandomGoalGenerator(grid_map, obstacle_search_distance_in_meters)

        elif goal_generator_type == "GoalReader":
            if not self.get_parameter("experience_path").value:
                self.get_logger().info("Experience path is not given. Returning..")
                sys.exit(1)

            file_path = self.get_parameter("experience_path").value
            goal_generator = GoalReader(file_path)
        else:
            self.get_logger().info("Invalid goal generator specified. Returning...")
            sys.exit(1)
        return goal_generator


def main(args=None):
    rclpy.init(args=args)
    set_goal = SetNavigationGoal()

    # Spin in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(set_goal, ), daemon=True)
    thread.start()

    set_goal.send_goal()
    
    thread.join()
    set_goal.destroy_node()
    # Only shutdown if context is still initialized (callbacks may have already shut it down)
    try:
        rclpy.shutdown()
    except RuntimeError:
        # Context was already shut down, which is fine
        pass


if __name__ == "__main__":
    main()
