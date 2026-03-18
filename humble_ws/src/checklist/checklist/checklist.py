# Copyright 2025 ReSim, Inc.
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import time
import rclpy
from rclpy.node import Node
from pathlib import Path
from rclpy.task import Future
from geometry_msgs.msg import Pose, Twist, Accel
from std_msgs.msg import Header
from simulation_interfaces.msg import Result, SimulationState, EntityState
from simulation_interfaces.srv import GetSimulationState, SetSimulationState, SetEntityState, LoadWorld

READY_PATH = Path("/tmp/isaac_ready")


class Checklist(Node):
    def __init__(self, future):
        super().__init__("checklist")
        self.future = future
        self.declare_parameter("initial_pose", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.declare_parameter("isaacsim_entity", "")
        self.declare_parameter("world_uri", "")

        self.initial_pose = list(self.get_parameter("initial_pose").get_parameter_value().double_array_value)
        self.isaacsim_entity = self.get_parameter("isaacsim_entity").get_parameter_value().string_value
        self.world_uri = self.get_parameter("world_uri").get_parameter_value().string_value

        if len(self.initial_pose) != 7:
            self.get_logger().error("initial_pose must contain exactly 7 values [x, y, z, qw, qx, qy, qz]")
            self.future.set_result(False)
            return
        if not self.isaacsim_entity:
            self.get_logger().error("isaacsim_entity parameter is required")
            self.future.set_result(False)
            return

        self.get_sim_state_client = self.create_client(GetSimulationState, "/get_simulation_state")
        self.set_sim_state_client = self.create_client(SetSimulationState, "/set_simulation_state")
        self.set_entity_state_client = self.create_client(SetEntityState, "/set_entity_state")
        self.load_world_client = self.create_client(LoadWorld, "/load_world")

        if not self._wait_for_services():
            self.future.set_result(False)
            return

        if not self._wait_until_stopped_or_paused():
            self.future.set_result(False)
            return

        if self.world_uri and not self._load_world():
            self.future.set_result(False)
            return

        if not self._set_initial_entity_pose():
            self.future.set_result(False)
            return

        self.get_logger().info("Checklist finished: world loaded and robot pose set. Ready for Nav2 startup.")
        READY_PATH.touch()
        self.future.set_result(True)

    def _wait_for_services(self) -> bool:
        clients = [
            ("LoadWorld", self.load_world_client),
            ("GetSimulationState", self.get_sim_state_client),
            ("SetSimulationState", self.set_sim_state_client),
            ("SetEntityState", self.set_entity_state_client),
        ]
        for service_name, client in clients:
            self.get_logger().info(f"Waiting for /isaacsim/{service_name} service...")
            if not client.wait_for_service(timeout_sec=30.0):
                self.get_logger().error(f"/isaacsim/{service_name} service not available")
                return False
        self.get_logger().info("All services available.")
        return True

    def _wait_until_stopped_or_paused(self) -> bool:
        # Simulator may come up playing or still initializing; wait until stable.
        max_attempts = 30
        for _ in range(max_attempts):
            state = self._get_simulation_state()
            if state is None:
                self.get_logger().warn("GetSimulationState failed, retrying...")
                continue

            if state.state in (SimulationState.STATE_STOPPED, SimulationState.STATE_PAUSED):
                self.get_logger().info("Isaac Sim is in a stable state for pose setting.")
                return True

            if state.state == SimulationState.STATE_PLAYING:
                self.get_logger().info("Isaac Sim is playing; requesting pause for pose update.")
                if not self._set_simulation_state(SimulationState.STATE_PAUSED):
                    return False
                continue

            if state.state == SimulationState.STATE_QUITTING:
                self.get_logger().error("Isaac Sim is quitting; cannot continue checklist.")
                return False
            
            self.get_logger().info(f"Isaac Sim state: {state}")

        self.get_logger().error("Timed out waiting for Isaac Sim to reach STOPPED/PAUSED.")
        return False

    def _get_simulation_state(self):
        request = GetSimulationState.Request()
        self.get_logger().info("Getting Isaac Sim state...")
        future = self.get_sim_state_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        response = future.result()
        if response is None:
            return None
        if response.result.result != Result.RESULT_OK:
            self.get_logger().warn(
                f"GetSimulationState failed ({response.result.result}): {response.result.error_message}"
            )
            return None
        return response.state

    def _set_simulation_state(self, target_state: int) -> bool:
        request = SetSimulationState.Request()
        request.state.state = target_state
        future = self.set_sim_state_client.call_async(request)
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
        return True

    def _set_initial_entity_pose(self) -> bool:
        request = SetEntityState.Request()
        request.entity = self.isaacsim_entity
        request.state = EntityState()
        request.state.header = Header()
        request.state.header.frame_id = "world"
        request.state.pose = Pose()
        request.state.pose.position.x = float(self.initial_pose[0])
        request.state.pose.position.y = float(self.initial_pose[1])
        request.state.pose.position.z = float(self.initial_pose[2])
        request.state.pose.orientation.w = float(self.initial_pose[3])
        request.state.pose.orientation.x = float(self.initial_pose[4])
        request.state.pose.orientation.y = float(self.initial_pose[5])
        request.state.pose.orientation.z = float(self.initial_pose[6])
        request.state.twist = Twist()
        request.state.acceleration = Accel()

        self.get_logger().info(
            f"Setting Isaac Sim entity {self.isaacsim_entity} initial pose to {self.initial_pose}"
        )
        future = self.set_entity_state_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is None:
            self.get_logger().error("SetEntityState service call failed.")
            return False
        if response.result.result != Result.RESULT_OK:
            self.get_logger().error(
                f"SetEntityState failed ({response.result.result}): {response.result.error_message}"
            )
            return False
        return True

    def _load_world(self) -> bool:
        self.get_logger().info(f"Loading Isaac Sim world from URI: {self.world_uri}")
        deadline = time.monotonic() + 60.0
        attempt = 1
        while True:
            request = LoadWorld.Request()
            request.uri = self.world_uri

            future = self.load_world_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            response = future.result()
            if response is not None and response.result.result == Result.RESULT_OK:
                return True

            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                if response is None:
                    self.get_logger().error("LoadWorld service call failed.")
                else:
                    self.get_logger().error(
                        f"LoadWorld failed ({response.result.result}): {response.result.error_message}"
                    )
                return False

            self.get_logger().warn(
                f"LoadWorld attempt {attempt} failed, retrying in 10s "
                f"({remaining:.1f}s remaining)."
            )
            time.sleep(min(10.0, remaining))
            attempt += 1

def main(args=None):
    rclpy.init(args=args)
    future = Future()
    node = Checklist(future)
    rclpy.spin_until_future_complete(node, future)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
