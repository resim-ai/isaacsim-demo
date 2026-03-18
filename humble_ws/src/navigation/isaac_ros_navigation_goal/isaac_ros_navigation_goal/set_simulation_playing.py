#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import sys

import rclpy
from rclpy.node import Node
from simulation_interfaces.msg import Result, SimulationState
from simulation_interfaces.srv import SetSimulationState


class SetSimulationPlaying(Node):
    def __init__(self):
        super().__init__("set_simulation_playing")
        self.client = self.create_client(SetSimulationState, "/set_simulation_state")

    def run(self) -> bool:
        if not self.client.wait_for_service(timeout_sec=30.0):
            self.get_logger().error("/set_simulation_state service not available")
            return False

        request = SetSimulationState.Request()
        request.state.state = SimulationState.STATE_PLAYING
        self.get_logger().info("Requesting Isaac Sim STATE_PLAYING.")
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is None:
            self.get_logger().error("SetSimulationState service call failed.")
            return False

        if response.result.result not in (
            Result.RESULT_OK,
            SetSimulationState.Response.ALREADY_IN_TARGET_STATE,
        ):
            self.get_logger().error(
                f"SetSimulationState failed ({response.result.result}): {response.result.error_message}"
            )
            return False

        self.get_logger().info("Isaac Sim is now PLAYING.")
        return True


def main(args=None):
    rclpy.init(args=args)
    node = SetSimulationPlaying()
    ok = node.run()
    node.destroy_node()
    rclpy.shutdown()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
