from pathlib import Path
from typing import Optional
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, PoseStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from builtin_interfaces.msg import Time as MsgTime
from custom_message.msg import GoalStatus
from resim.metrics.python.emissions import emit
from tf2_ros import Buffer, TransformListener, Duration
import tf2_geometry_msgs
import os
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

class MetricsEmitter(Node):
    def __init__(self):
        super().__init__('metrics_emitter')
        self.get_logger().info('Metrics emitter node initialized')

        os.makedirs("/tmp/resim/outputs/ignore", exist_ok=True)
        self.emissions_handle = Path('/tmp/resim/outputs/ignore/emissions.ndjson').open('a')

        # build transform buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # data for distance to goal metric
        callback_group = MutuallyExclusiveCallbackGroup() # ensure callbacks are not called concurrently
        self.goal_subscriber = self.create_subscription(PoseStamped, '/goal', self.goal_callback, 10, callback_group=callback_group)
        self.goal_status_subscriber = self.create_subscription(GoalStatus, '/goal/status', self.goal_status_callback, 10, callback_group=callback_group)
        self.chassis_odom_subscriber = self.create_subscription(Odometry, '/chassis/odom', self.chassis_odom_callback, 10, callback_group=callback_group)
        
        # data for pose difference metric
        self.amcl_subscriber = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_callback, 10, callback_group=callback_group)

        self.first_goal_received_time: Optional[Time] = None
        self.prev_goal_received_time: Optional[Time] = None
        
        # State for distance to goal metric
        self.current_goal: Optional[PoseStamped] = None
        self.transformed_goal: Optional[Pose] = None
        
        # State for goal completion tracking
        self.goal_count = 0
        self.completed_goals = -1
        
        # State for pose difference metric
        self.latest_odom: Optional[Odometry] = None
        
        # Rate limiting for odometry processing (10Hz = 0.1 seconds)
        self.odom_processing_rate = 0.1  # seconds between processing
        self.last_odom_processed_time: Optional[Time] = None

    def __del__(self):
        self.emissions_handle.close()

    def get_relative_timestamp(self, msg_time: Optional[MsgTime] = None) -> Optional[int]:
        """
        Get the current timestamp relative to the first goal received time.
        
        Args:
            current_time: The current time (defaults to now)
            
        Returns:
            Relative timestamp in nanoseconds, or None if no first goal time is set
        """
        if self.first_goal_received_time is None:
            return None

        current_time: Time
        if msg_time is not None:
            current_time = Time.from_msg(msg_time)
        else:
            current_time = self.get_clock().now()
            
        return current_time.nanoseconds - self.first_goal_received_time.nanoseconds

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
        
        self.get_logger().info(f"New goal received (#{self.goal_count}) at time {msg.header.stamp.sec}.{msg.header.stamp.nanosec} in frame: {msg.header.frame_id}")
        
        # Store the first goal received time
        new_goal_time = Time.from_msg(msg.header.stamp)
        if self.first_goal_received_time is None:
            self.first_goal_received_time = new_goal_time

        self.prev_goal_received_time = new_goal_time
            
        # Transform goal from its frame (typically "map") to "odom" frame
        try:            
            # Get transform from goal frame to odom frame
            transform = self.tf_buffer.lookup_transform(
                "odom",  # target frame
                msg.header.frame_id,  # source frame (typically "map")
                new_goal_time,
                timeout=Duration(seconds=1)
            )
            
            # Transform the pose
            pose_stamped_transformed = tf2_geometry_msgs.do_transform_pose_stamped(msg, transform)
            self.transformed_goal = pose_stamped_transformed.pose
            
            self.get_logger().info(f"Goal transformed to odom frame: x={self.transformed_goal.position.x:.2f}, y={self.transformed_goal.position.y:.2f}")
            
        except Exception as ex:
            self.get_logger().warn(f'Could not transform goal to odom frame: {ex}')
            self.transformed_goal = None

    def goal_status_callback(self, msg: GoalStatus):
        """Handle goal status messages."""
        self.get_logger().info(f"Received goal status message at {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
        self.completed_goals += 1

        # A NEW_GOAL message is sent for the first goal, so we skip
        if self.completed_goals == 0:
            return
        
        # Calculate timestamp relative to first goal received time
        relative_timestamp = self.get_relative_timestamp(msg.header.stamp)
        if relative_timestamp is None:
            return
        
        # Emit goal reached event for any goal completion
        emit('goal_reached', {
            'name': f'Goal {self.completed_goals} Reached',
            'description': f'Robot successfully reached goal number {self.completed_goals}',
            'status': 'PASSED',
            'tags': ['navigation']
        }, event=True, timestamp=relative_timestamp, file=self.emissions_handle)

        if self.prev_goal_received_time is not None and self.first_goal_received_time is not None:
            time_to_goal_seconds = (relative_timestamp - (self.prev_goal_received_time.nanoseconds - self.first_goal_received_time.nanoseconds)) / 1e9
        else:
            self.get_logger().warn("Cannot calculate time to goal: missing timestamp data")
            return

        emit('time_to_goal', {
            'time_s': time_to_goal_seconds,
            'goal_name': f'Goal {self.completed_goals}'
        }, timestamp=relative_timestamp, file=self.emissions_handle)
        
        self.get_logger().info(f"Goal {self.completed_goals} completed with status: {msg.status}")
        
        if msg.status == "COMPLETE":
            self.get_logger().info(f"Final goal reached! Total time: {relative_timestamp / 1e9:.2f}s")

    def chassis_odom_callback(self, msg: Odometry):
        """Handle odometry messages and calculate distance to goal at 10Hz."""
        # Store latest odometry for pose difference calculation
        self.latest_odom = msg
        
        # Only calculate distance if we have a transformed goal
        if self.transformed_goal is None or self.first_goal_received_time is None or self.prev_goal_received_time is None:
            return

        try:
            current_time = Time.from_msg(msg.header.stamp)
            current_time_ns = current_time.nanoseconds
            
            # Rate limiting: only process at 10Hz (every 0.1 seconds)
            if self.last_odom_processed_time is not None:
                time_since_last = (current_time - self.last_odom_processed_time).nanoseconds / 1e9
                if time_since_last < self.odom_processing_rate:
                    return  # Skip this message to maintain 10Hz processing rate
            
            # messages may be out of order, so we skip if the current time is before the previous goal received time
            if current_time_ns < self.prev_goal_received_time.nanoseconds:
                self.get_logger().warn(f"Skipping odom message to prevent time travel.")
                return

            # Calculate distance using odom coordinates (2D distance)
            dx = self.transformed_goal.position.x - msg.pose.pose.position.x
            dy = self.transformed_goal.position.y - msg.pose.pose.position.y
            distance = (dx * dx + dy * dy) ** 0.5
            
            # Calculate timestamp relative to first goal received time
            relative_timestamp = self.get_relative_timestamp(msg.header.stamp)
            if relative_timestamp is None:
                self.get_logger().warn(f"Could not calculate timestamp relative to first goal received time.")
                return
            
            # Emit the distance metric
            emit('goal_distance_1', {'distance_m': distance, 'goal_name': f'Goal {self.goal_count}'}, timestamp=relative_timestamp, file=self.emissions_handle)
            
            # Update the last processed time for rate limiting
            self.last_odom_processed_time = current_time
            
            self.get_logger().debug(f"Distance to goal: {distance:.2f}m (processed at 10Hz)")
            
        except Exception as ex:
            self.get_logger().warn(f'Could not calculate distance to goal: {ex}')

    def amcl_callback(self, msg: PoseWithCovarianceStamped):
        """Handle AMCL pose messages and calculate pose difference."""
        # Only calculate pose difference if we have latest odometry and first goal time is set
        if self.latest_odom is None or self.first_goal_received_time is None:
            return

        try:
            # Get AMCL pose position (already in map frame)
            amcl_pos = msg.pose.pose.position
            
            # Transform odometry pose to map frame
            odom_pose = self.latest_odom.pose.pose
            odom_header = self.latest_odom.header
            
            # Get transform from odom frame to map frame at AMCL timestamp
            transform = self.tf_buffer.lookup_transform(
                msg.header.frame_id,  # target frame (typically "map")
                odom_header.frame_id,  # source frame (typically "odom")
                Time.from_msg(msg.header.stamp),
                timeout=Duration(seconds=1)
            )
            
            # Transform the odometry pose to map frame
            odom_pose_stamped = PoseStamped()
            odom_pose_stamped.header = odom_header
            odom_pose_stamped.pose = odom_pose
            odom_in_map = tf2_geometry_msgs.do_transform_pose_stamped(odom_pose_stamped, transform)
            odom_pos = odom_in_map.pose.position
            
            # Calculate 3D position difference
            pos_diff = ((amcl_pos.x - odom_pos.x) ** 2 + 
                       (amcl_pos.y - odom_pos.y) ** 2 + 
                       (amcl_pos.z - odom_pos.z) ** 2) ** 0.5
            
            # Calculate timestamp relative to first goal received time
            relative_timestamp = self.get_relative_timestamp(msg.header.stamp)
            if relative_timestamp is None:
                self.get_logger().warn(f"Could not calculate timestamp relative to first goal received time.")
                return
            
            # Emit the pose difference metric
            emit('pose_difference', {
                'position_diff_m': pos_diff
            }, timestamp=relative_timestamp, file=self.emissions_handle)
            
            self.get_logger().debug(f"Pose difference: {pos_diff:.3f}m")
            
        except Exception as ex:
            self.get_logger().warn(f'Could not calculate pose difference: {ex}')


def main(args=None):
    rclpy.init(args=args)
    metrics_emitter = MetricsEmitter()
    rclpy.spin(metrics_emitter)
    metrics_emitter.destroy_node()
    rclpy.shutdown()