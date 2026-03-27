#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import sys
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node


class PublishInitialPose(Node):
    def __init__(self):
        super().__init__("publish_initial_pose")
        self.declare_parameter("initial_pose", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("wait_for_subscribers_timeout_sec", 30.0)
        self.declare_parameter("publish_repetitions", 1)
        self.declare_parameter("publish_period_sec", 0.2)
        self.publisher = self.create_publisher(PoseWithCovarianceStamped, "initialpose", 10)

    def run(self) -> bool:
        initial_pose = list(self.get_parameter("initial_pose").get_parameter_value().double_array_value)
        if len(initial_pose) != 7:
            self.get_logger().error("initial_pose must contain exactly 7 values [x, y, z, qw, qx, qy, qz]")
            return False

        timeout_sec = float(
            self.get_parameter("wait_for_subscribers_timeout_sec").get_parameter_value().double_value
        )
        start_time = time.monotonic()
        while self.publisher.get_subscription_count() == 0 and (time.monotonic() - start_time) < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.publisher.get_subscription_count() == 0:
            self.get_logger().error(
                "No /initialpose subscribers detected before timeout; refusing to publish."
            )
            return False

        self.get_logger().info("Detected /initialpose subscriber(s), publishing initial pose.")

        frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        repetitions = int(self.get_parameter("publish_repetitions").get_parameter_value().integer_value)
        publish_period_sec = float(self.get_parameter("publish_period_sec").get_parameter_value().double_value)
        repetitions = max(1, repetitions)

        for _ in range(repetitions):
            msg = PoseWithCovarianceStamped()
            msg.header.frame_id = frame_id
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.pose.position.x = float(initial_pose[0])
            msg.pose.pose.position.y = float(initial_pose[1])
            msg.pose.pose.position.z = float(initial_pose[2])
            msg.pose.pose.orientation.w = float(initial_pose[3])
            msg.pose.pose.orientation.x = float(initial_pose[4])
            msg.pose.pose.orientation.y = float(initial_pose[5])
            msg.pose.pose.orientation.z = float(initial_pose[6])
            self.publisher.publish(msg)
            time.sleep(publish_period_sec)

        self.get_logger().info(f"Published /initialpose {repetitions} time(s).")
        return True


def main(args=None):
    rclpy.init(args=args)
    node = PublishInitialPose()
    ok = node.run()
    node.destroy_node()
    rclpy.shutdown()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
