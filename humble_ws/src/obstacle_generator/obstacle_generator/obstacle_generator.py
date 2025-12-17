# Copyright 2025 ReSim, Inc.
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import rclpy
from rclpy.node import Node
from rclpy.task import Future
import numpy as np
import math
import re
from geometry_msgs.msg import PoseStamped, Quaternion, Pose, Twist, Accel
from std_msgs.msg import Header
from simulation_interfaces.srv import SpawnEntity, GetEntities, SetEntityState
from simulation_interfaces.msg import Result, EntityState, EntityFilters
from isaac_ros_navigation_goal.obstacle_map import GridMap
from obstacle_generator.obstacle_library import OBSTACLES


class ObstacleGenerator(Node):
    def __init__(self, future):
        super().__init__("obstacle_generator")
        self.future = future  # resolve this when the obstacle generator is complete
        
        # Declare parameters
        self.declare_parameter("map_yaml_path", "")
        self.declare_parameter("goal_text_file_path", "")
        self.declare_parameter("seed", -1)
        self.declare_parameter("num_obstacles", 20)
        self.declare_parameter("initial_pose", [-6.4, -1.04, 0.0, 0.0, 0.0, 0.99, 0.02])
        
        map_yaml_path = self.get_parameter("map_yaml_path").value
        goal_text_file_path = self.get_parameter("goal_text_file_path").value
        seed = self.get_parameter("seed").get_parameter_value().integer_value
        num_obstacles = self.get_parameter("num_obstacles").get_parameter_value().integer_value
        initial_pose = self.get_parameter("initial_pose").get_parameter_value().double_array_value
        
        # Store initial pose (x, y) for collision checking
        self.initial_pose_xy = (initial_pose[0], initial_pose[1]) if len(initial_pose) >= 2 else None
        
        if not map_yaml_path:
            self.get_logger().error("map_yaml_path parameter is required")
            future.set_result(False)
            return
        
        if not goal_text_file_path:
            self.get_logger().error("goal_text_file_path parameter is required")
            future.set_result(False)
            return
        
        # Set random seed if provided
        if seed >= 0:
            np.random.seed(seed)
            self.get_logger().info(f"Using provided seed: {seed}")
        else:
            self.get_logger().info("No seed provided, using random generation")
        
        # Initialize GridMap
        try:
            self.get_logger().info(f"Loading map from {map_yaml_path}")
            self.grid_map = GridMap(map_yaml_path)
        except Exception as e:
            self.get_logger().error(f"Failed to load map: {e}")
            future.set_result(False)
            return
        
        # Load goals
        self.get_logger().info(f"Loading goals from {goal_text_file_path}")
        self.goals = self._load_goals(goal_text_file_path)
        self.get_logger().info(f"Loaded {len(self.goals)} goals")
        
        # Create service clients
        self.spawn_client = self.create_client(SpawnEntity, "/isaacsim/SpawnEntity")
        self.get_entities_client = self.create_client(GetEntities, "/isaacsim/GetEntities")
        self.set_entity_state_client = self.create_client(SetEntityState, "/isaacsim/SetEntityState")
        
        # Wait for services to be available
        self.get_logger().info("Waiting for services...")
        if not self.spawn_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("SpawnEntity service not available")
            future.set_result(False)
            return
        if not self.get_entities_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("GetEntities service not available")
            future.set_result(False)
            return
        if not self.set_entity_state_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("SetEntityState service not available")
            future.set_result(False)
            return
        
        self.get_logger().info(f"Obstacle generator initialized (target: {num_obstacles} obstacles)")
        
        # Spawn obstacles
        try:
            self._spawn_obstacles(num_obstacles)
            # Mark as complete
            future.set_result(True)
        except Exception as e:
            self.get_logger().error(f"Error during obstacle spawning: {e}")
            future.set_result(False)
    
    def _load_goals(self, file_path):
        """Load goals from text file."""
        goals = []
        try:
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        values = list(map(float, line.split()))
                        if len(values) >= 2:
                            # Store only x, y coordinates
                            goals.append((values[0], values[1]))
        except Exception as e:
            self.get_logger().error(f"Failed to load goals: {e}")
        return goals
    
    def _yaw_to_quaternion(self, yaw):
        """Convert yaw angle (in radians) to quaternion."""
        q = Quaternion()
        q.w = math.cos(yaw / 2.0)
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        return q
    
    def _distance(self, pos1, pos2):
        """Calculate Euclidean distance between two 2D positions."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _is_valid_position(self, x, y, spawned_obstacles, min_distance_to_goals=2.0, min_distance_to_obstacles=2.5, min_distance_to_initial_pose=4.0):
        """Check if a position is valid for spawning an obstacle."""
        # Check static map collision (using 1.5m radius for 2x2m obstacle)
        if not self.grid_map.is_valid_pose((x, y), distance=1.5):
            return False
        
        # Check distance to initial pose
        if self.initial_pose_xy is not None:
            if self._distance((x, y), self.initial_pose_xy) < min_distance_to_initial_pose:
                return False
        
        # Check distance to goals
        for goal_x, goal_y in self.goals:
            if self._distance((x, y), (goal_x, goal_y)) < min_distance_to_goals:
                return False
        
        # Check distance to previously spawned obstacles
        for obstacle_x, obstacle_y in spawned_obstacles:
            if self._distance((x, y), (obstacle_x, obstacle_y)) < min_distance_to_obstacles:
                return False
        
        return True
    
    def _generate_random_position(self, spawned_obstacles, max_trials=1000):
        """Generate a random valid position for an obstacle."""
        map_range = self.grid_map.get_range()
        x_min, x_max = map_range[0]
        y_min, y_max = map_range[1]
        
        for _ in range(max_trials):
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            
            if self._is_valid_position(x, y, spawned_obstacles):
                return (x, y)
        
        return None
    
    def _get_existing_obstacles(self):
        """Get list of existing obstacle entities."""
        request = GetEntities.Request()
        request.filters = EntityFilters()
        request.filters.filter = "Obstacle[0-9]+$"  # Match Obstacle1, Obstacle2, etc.
        
        # Call service
        future = self.get_entities_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        response = future.result()
        if response is None:
            self.get_logger().warn("Failed to get existing entities: service call failed")
            return []
        
        if response.result.result != Result.RESULT_OK:
            self.get_logger().warn(f"Failed to get existing entities: {response.result.error_message}")
            return []
        
        # Sort obstacles by name to ensure consistent ordering
        obstacles = sorted(response.entities)
        print(obstacles)
        return obstacles
    
    def _move_obstacle(self, entity_name, x, y, yaw, z_offset=0.0):
        """Move an existing obstacle to a new position using SetEntityState."""
        request = SetEntityState.Request()
        request.entity = entity_name
        
        # Create EntityState with new pose
        request.state = EntityState()
        request.state.header = Header()
        request.state.header.frame_id = "world"
        
        request.state.pose = Pose()
        request.state.pose.position.x = float(x)
        request.state.pose.position.y = float(y)
        request.state.pose.position.z = float(z_offset)
        request.state.pose.orientation = self._yaw_to_quaternion(yaw)
        
        # Set twist and acceleration to zero for static object
        request.state.twist = Twist()
        request.state.acceleration = Accel()
        
        # Call service
        future = self.set_entity_state_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        response = future.result()
        if response is None:
            self.get_logger().error(f"Failed to move {entity_name}: service call failed")
            return False
        
        if response.result.result != Result.RESULT_OK:
            self.get_logger().error(f"Failed to move {entity_name}: {response.result.error_message}")
            return False
        
        self.get_logger().info(f"Successfully moved {entity_name} to ({x:.2f}, {y:.2f}, {z_offset:.2f}) with yaw {yaw:.2f}")
        return True
    
    def _spawn_obstacle(self, name, x, y, yaw, uri, z_offset=0.0):
        """Spawn a single obstacle using the SpawnEntity service."""
        request = SpawnEntity.Request()
        request.name = name
        request.allow_renaming = False
        request.uri = uri
        request.resource_string = ""
        request.entity_namespace = ""
        
        # Set pose
        request.initial_pose = PoseStamped()
        request.initial_pose.header.frame_id = "world"
        request.initial_pose.pose.position.x = float(x)
        request.initial_pose.pose.position.y = float(y)
        request.initial_pose.pose.position.z = float(z_offset)
        request.initial_pose.pose.orientation = self._yaw_to_quaternion(yaw)
        
        # Call service
        future = self.spawn_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        response = future.result()
        if response is None:
            self.get_logger().error(f"Failed to spawn {name}: service call failed")
            return False
        
        if response.result.result != Result.RESULT_OK:
            self.get_logger().error(f"Failed to spawn {name}: {response.result.error_message}")
            return False
        
        self.get_logger().info(f"Successfully spawned {name} at ({x:.2f}, {y:.2f}, {z_offset:.2f}) with yaw {yaw:.2f}")
        return True
    
    def _spawn_obstacles(self, target_count):
        """Spawn or move obstacles at random valid locations. Keeps retrying until target_count are successfully placed."""
        # Build weighted distribution list from OBSTACLES based on weight field
        weighted_obstacles = []
        for obstacle in OBSTACLES:
            weight = obstacle.get("weight", 1)  # Default weight of 1 if not specified
            for _ in range(weight):
                weighted_obstacles.append(obstacle)
        
        # Pre-generate object types using round-robin on weighted list to ensure stable assignment regardless of seed
        obstacle_types = []
        for i in range(target_count):
            obstacle_type = weighted_obstacles[i % len(weighted_obstacles)]
            obstacle_types.append(obstacle_type)
        
        # Check for existing obstacles
        existing_obstacles_list = self._get_existing_obstacles()
        self.get_logger().info(f"Found {len(existing_obstacles_list)} existing obstacles")
        
        # Create a mapping from obstacle index to existing obstacle name
        # Extract index from entity names (e.g., "/Obstacle1" -> 1, "Obstacle10" -> 10)
        existing_obstacles_map = {}
        for entity_name in existing_obstacles_list:
            # Remove leading slash if present and extract number
            match = re.search(r'Obstacle(\d+)', entity_name)
            if match:
                idx = int(match.group(1))
                existing_obstacles_map[idx] = entity_name
        
        spawned_obstacles = []
        attempts = 0
        
        while len(spawned_obstacles) < target_count:
            obstacle_index = len(spawned_obstacles) + 1
            obstacle_name = f"Obstacle{obstacle_index}"
            obstacle_type = obstacle_types[obstacle_index - 1]
            self.get_logger().info(f"Generating position for {obstacle_name} ({obstacle_type['name']})...")
            
            # Generate random position (keep retrying until valid)
            position = self._generate_random_position(spawned_obstacles)
            if position is None:
                self.get_logger().warn(f"Failed to generate valid position for {obstacle_name} after max trials, retrying...")
                continue
            
            x, y = position
            
            # Generate random yaw
            yaw = np.random.uniform(0, 2 * math.pi)
            
            # Check if this obstacle already exists
            if obstacle_index in existing_obstacles_map:
                # Move existing obstacle - use the correct type's z_offset for this index
                existing_name = existing_obstacles_map[obstacle_index]
                if self._move_obstacle(existing_name, x, y, yaw, obstacle_type['z_offset']):
                    spawned_obstacles.append((x, y))
                    attempts = 0  # Reset attempts on success
                    self.get_logger().info(f"Successfully moved {existing_name} ({len(spawned_obstacles)}/{target_count} complete)")
                else:
                    self.get_logger().warn(f"Failed to move {existing_name}, retrying...")
                    attempts += 1
                    if attempts > 25:
                        self.get_logger().error(f"Failed to move {existing_name} after 25 attempts, giving up.")
                        break
            else:
                # Spawn new obstacle
                if self._spawn_obstacle(obstacle_name, x, y, yaw, obstacle_type['uri'], obstacle_type['z_offset']):
                    spawned_obstacles.append((x, y))
                    attempts = 0  # Reset attempts on success
                    self.get_logger().info(f"Successfully spawned {obstacle_name} ({len(spawned_obstacles)}/{target_count} complete)")
                else:
                    self.get_logger().warn(f"Failed to spawn {obstacle_name}, retrying...")
                    attempts += 1
                    if attempts > 25:
                        self.get_logger().error(f"Failed to spawn {obstacle_name} after 25 attempts, giving up.")
                        break
        
        moved_count = min(len(existing_obstacles_map), target_count)
        spawned_count = max(0, target_count - len(existing_obstacles_map))
        self.get_logger().info(f"Successfully placed all {len(spawned_obstacles)} obstacles ({moved_count} moved, {spawned_count} spawned)")


def main(args=None):
    rclpy.init(args=args)
    future = Future()
    node = ObstacleGenerator(future)
    
    # Spin until future is complete (which happens when obstacles are spawned)
    rclpy.spin_until_future_complete(node, future)
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
