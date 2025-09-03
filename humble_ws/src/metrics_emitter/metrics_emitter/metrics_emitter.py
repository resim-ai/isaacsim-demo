from pathlib import Path
from typing import Optional
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, Pose
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.time import Time as RclpyTime
from resim.metrics.python.emissions import emit
from example_interfaces.msg import String
from tf2_ros import Buffer, TransformListener, Duration
import tf2_geometry_msgs
import os
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

class MetricsEmitter(Node):
    def __init__(self):
        super().__init__('metrics_emitter')
        self.get_logger().info('Metrics emitter node initialized')

        os.makedirs("/tmp/resim/outputs", exist_ok=True)
        self.emissions_handle = Path('/tmp/resim/outputs/emissions.ndjson').open('a')

        # build transform buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # data for distance to goal metric
        self.goal_subscriber = self.create_subscription(PoseStamped, '/goal', self.goal_callback, 10)
        self.goal_status_subscriber = self.create_subscription(String, '/goal/status', self.goal_status_callback, 10)
        self.chassis_odom_subscriber = self.create_subscription(Odometry, '/chassis/odom', self.chassis_odom_callback, 1)

        self.first_goal_received_time: Optional[Time] = None
        
        # State for distance to goal metric
        self.current_goal: Optional[PoseStamped] = None
        self.transformed_goal: Optional[Pose] = None
        
        # State for goal completion tracking
        self.goal_count = 0
        self.completed_goals = 0

    def __del__(self):
        self.emissions_handle.close()

    def get_transform(self, target_frame: str, source_frame: str, time=None):
        """
        Get transform between two coordinate frames.
        
        Args:
            target_frame: The target coordinate frame
            source_frame: The source coordinate frame
            time: The time for the transform (defaults to now)
            
        Returns:
            The transform or None if not available
        """
        try:
            if time is None:
                time = self.get_clock().now()
            
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, time, timeout=Duration(seconds=1)
            )
            return transform
        except Exception as ex:
            self.get_logger().warn(f'Could not transform {source_frame} to {target_frame}: {ex}')
            return None

    def goal_callback(self, msg: PoseStamped):
        """Handle new goal messages."""
        self.current_goal = msg
        self.goal_count += 1
        
        self.get_logger().info(f"New goal received (#{self.goal_count}) in frame: {msg.header.frame_id}")
        
        # Store the first goal received time
        if self.first_goal_received_time is None:
            self.first_goal_received_time = msg.header.stamp
            
        # Transform goal from its frame (typically "map") to "odom" frame
        try:
            # Create time from message header
            transform_time = RclpyTime(
                seconds=msg.header.stamp.sec,
                nanoseconds=msg.header.stamp.nanosec
            )
            
            # Get transform from goal frame to odom frame
            transform = self.tf_buffer.lookup_transform(
                "odom",  # target frame
                msg.header.frame_id,  # source frame (typically "map")
                transform_time,
                timeout=Duration(seconds=1)
            )
            
            # Transform the pose
            pose_stamped_transformed = tf2_geometry_msgs.do_transform_pose_stamped(msg, transform)
            self.transformed_goal = pose_stamped_transformed.pose
            
            self.get_logger().info(f"Goal transformed to odom frame: x={self.transformed_goal.position.x:.2f}, y={self.transformed_goal.position.y:.2f}")
            
        except Exception as ex:
            self.get_logger().warn(f'Could not transform goal to odom frame: {ex}')
            self.transformed_goal = None

    def goal_status_callback(self, msg: String):
        """Handle goal status messages."""
        self.get_logger().info("Received goal status message.")
        if self.first_goal_received_time is None:
            return

        self.completed_goals += 1
        
        # Calculate timestamp relative to first goal received time
        current_time = self.get_clock().now()
        current_time_ns = current_time.nanoseconds
        first_goal_time_ns = self.first_goal_received_time.sec * int(1e9) + self.first_goal_received_time.nanosec
        relative_timestamp = current_time_ns - first_goal_time_ns
        
        # Emit goal reached event for any goal completion
        emit('goal_reached', {
            'name': f'Goal {self.completed_goals} Reached',
            'description': f'Robot successfully reached goal number {self.completed_goals}',
            'status': 'PASSED',
            'tags': [f'goal_{self.completed_goals}', 'navigation']
        }, event=True, timestamp=relative_timestamp, file=self.emissions_handle)
        
        self.get_logger().info(f"Goal {self.completed_goals} completed with status: {msg.data}")
        
        # Only emit time_to_goal for the final goal (COMPLETE status)
        if msg.data == "COMPLETE":
            time_to_goal_seconds = relative_timestamp / 1e9
            
            emit('time_to_goal', {
                'time_s': time_to_goal_seconds
            }, timestamp=relative_timestamp, file=self.emissions_handle)
            
            self.get_logger().info(f"Final goal reached! Total time: {time_to_goal_seconds:.2f}s")

    def chassis_odom_callback(self, msg: Odometry):
        """Handle odometry messages and calculate distance to goal."""
        # Only calculate distance if we have a transformed goal
        if self.transformed_goal is None or self.first_goal_received_time is None:
            return
            
        try:
            # Calculate distance using odom coordinates (2D distance)
            dx = self.transformed_goal.position.x - msg.pose.pose.position.x
            dy = self.transformed_goal.position.y - msg.pose.pose.position.y
            distance = (dx * dx + dy * dy) ** 0.5
            
            # Calculate timestamp relative to first goal received time
            current_time_ns = msg.header.stamp.sec * int(1e9) + msg.header.stamp.nanosec
            first_goal_time_ns = self.first_goal_received_time.sec * int(1e9) + self.first_goal_received_time.nanosec
            relative_timestamp = current_time_ns - first_goal_time_ns
            
            # Emit the distance metric
            emit('goal_distance', {'distance_m': distance}, timestamp=relative_timestamp, file=self.emissions_handle)
            
            self.get_logger().debug(f"Distance to goal: {distance:.2f}m")
            
        except Exception as ex:
            self.get_logger().warn(f'Could not calculate distance to goal: {ex}')


def main(args=None):
    rclpy.init(args=args)
    metrics_emitter = MetricsEmitter()
    rclpy.spin(metrics_emitter)
    metrics_emitter.destroy_node()
    rclpy.shutdown()