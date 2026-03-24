# Copyright 2025 ReSim, Inc.
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from simulation_interfaces.msg import Result, SimulationState
from simulation_interfaces.srv import SetSimulationState


READY_PATH = Path("/tmp/isaac_ready")

STATE_NAME_TO_ENUM = {
    "playing": SimulationState.STATE_PLAYING,
    "paused": SimulationState.STATE_PAUSED,
    "stopped": SimulationState.STATE_STOPPED,
}


class SetSimulationStateNode(Node):
    def __init__(self):
        super().__init__("set_simulation_state")
        self.client = self.create_client(SetSimulationState, "/set_simulation_state")
        self.declare_parameter("target_state", "playing")

    def run(self) -> bool:
        target_state_name = (
            self.get_parameter("target_state").get_parameter_value().string_value.lower()
        )
        target_state = STATE_NAME_TO_ENUM.get(target_state_name)
        if target_state is None:
            self.get_logger().error(
                "Invalid target_state '%s'. Valid values: %s"
                % (target_state_name, ", ".join(sorted(STATE_NAME_TO_ENUM.keys())))
            )
            return False

        if not self.client.wait_for_service(timeout_sec=30.0):
            self.get_logger().error("/set_simulation_state service not available")
            return False

        request = SetSimulationState.Request()
        request.state.state = target_state
        self.get_logger().info(f"Requesting Isaac Sim STATE_{target_state_name.upper()}.")
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

        self.get_logger().info(f"Isaac Sim is now {target_state_name.upper()}.")
        if target_state == SimulationState.STATE_PLAYING:
            READY_PATH.touch()
        return True


def main(args=None):
    rclpy.init(args=args)
    node = SetSimulationStateNode()
    ok = node.run()
    node.destroy_node()
    rclpy.shutdown()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
