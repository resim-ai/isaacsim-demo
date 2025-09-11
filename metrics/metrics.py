import argparse
from enum import Enum
import json
import logging
import math
import sys
from typing import Any, Sequence, cast
import uuid
from pathlib import Path
import cv2
import numpy as np
import imageio
import rclpy.serialization
from resim.metrics.python.emissions import emit
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image as ROSImage
from geometry_msgs.msg import PoseStamped, Pose
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer
import tf2_geometry_msgs
from rclpy.time import Time, Duration
from custom_message.msg import GoalStatus

import resim.metrics.fetch_job_metrics as fjm
from resim.metrics.fetch_all_pages import fetch_all_pages
from resim.metrics.fetch_job_metrics import fetch_job_metrics
from resim_python_client import AuthenticatedClient
from resim_python_client.api.batches import list_jobs


from resim.metrics.resim_style import resim_plotly_style, resim_colors as RESIM_COLORS, RESIM_TURQUOISE
from resim.metrics.proto.validate_metrics_proto import validate_job_metrics
from resim.metrics.python.metrics_utils import (
    ResimMetricsOutput,
    MetricImportance,
    MetricStatus,
    Timestamp,
)
from resim.metrics.python.metrics import ExternalFileMetricsData, SeriesMetricsData, ScalarMetric
from resim.metrics.python.metrics_writer import ResimMetricsWriter
from resim.transforms.python.quaternion import Quaternion
from resim.transforms.python.so3_python import SO3
from resim.metrics.python.unpack_metrics import UnpackedMetrics
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger("metrics.py")
logger.propagate = False
handler = logging.StreamHandler(stream=sys.stderr)
handler.setFormatter(logging.Formatter(fmt="%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)


DEFAULT_METRICS_PATH = "/tmp/resim/outputs/metrics.binproto"
DEFAULT_LOG_PATH = "/tmp/resim/inputs/logs/record_0.mcap"
DEFAULT_BATCH_METRICS_CONFIG_PATH = "/tmp/resim/inputs/batch_metrics_config.json"


class Topics(Enum):
    LOCAL_COSTMAP = "/local_costmap"
    LIDAR = "/hesai/pandar_points"


class FrameIds(Enum):
    BASE_LINK = "base_link"
    ODOM = "odom"
    MAP = "map"


def parse_args() -> argparse.Namespace:
    """Parses arguments."""

    parser = argparse.ArgumentParser()

    # Adding fields with default values
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help=f"Path to the output file (default: {DEFAULT_METRICS_PATH})",
    )

    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help=f"Path to the input log file (default: {DEFAULT_LOG_PATH})",
    )

    parser.add_argument(
        "--batch-metrics-config-path",
        type=Path,
        default=DEFAULT_BATCH_METRICS_CONFIG_PATH,
        help=(
            f"Path to the batch metrics configuration file (default: "
            f"{DEFAULT_BATCH_METRICS_CONFIG_PATH})"
        ),
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Set up logging
    logger.setLevel(getattr(logging, args.log_level))

    return args


def write_proto(writer, path: str) -> ResimMetricsOutput:
    """Write out the binproto for our metrics."""
    metrics_proto = writer.write()
    validate_job_metrics(metrics_proto.metrics_msg)
    # Known location where the runner looks for metrics
    with open(path, "wb") as f:
        f.write(metrics_proto.metrics_msg.SerializeToString())

    return metrics_proto


def read_messages(input_bag: str, topics: list[str]):
    """Read messages from an MCAP file using ROS 2.

    Args:
        input_bag: Path to the MCAP file
        topics: List of topics to read

    Yields:
        Tuple of (topic, message, timestamp) for each message
    """
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=input_bag, storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )

    topic_types = reader.get_all_topics_and_types()

    def typename(topic_name):
        for topic_type in topic_types:
            if topic_type.name == topic_name:
                return topic_type.type
        raise ValueError(f"topic {topic_name} not in bag")

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic not in topics:
            continue
        msg_type = get_message(typename(topic))
        msg = rclpy.serialization.deserialize_message(data, msg_type)
        yield topic, msg, timestamp


def seconds_array_to_timestamp(seconds: Sequence[float]) -> np.ndarray:
    return np.array(
        [Timestamp(secs=int(s), nanos=int((s - int(s)) * 1e9)) for s in seconds]
    )


class TransformManager:
    """Manages transform collection and normalization for metrics processing."""

    def __init__(self):
        """Initialize the transform manager with a long-lived buffer."""
        self.tf_buffer = Buffer(cache_time=Duration(seconds=pow(2, 31) - 1))
        self.seen_frame_pairs = set()  # Track which frame pairs we've seen

    def collect_transforms(self, input_bag: Path):
        """Collect all transforms from the bag file.

        Args:
            input_bag: Path to the MCAP file
        """
        logger.info("Collecting transforms...")

        for topic, msg, _ in read_messages(str(input_bag), ["/tf", "/tf_static"]):
            assert isinstance(msg, TFMessage)
            for transform in msg.transforms:
                # Only collect transforms between odom and map frames
                if not (
                    transform.header.frame_id in [e.value for e in FrameIds]
                    and transform.child_frame_id in [e.value for e in FrameIds]
                ):
                    continue

                key = (transform.header.frame_id, transform.child_frame_id)

                # Only normalize the first transform for each frame pair
                if key not in self.seen_frame_pairs:
                    self.seen_frame_pairs.add(key)
                    # Set the first transform's time to 0
                    transform.header.stamp.sec = 0
                    transform.header.stamp.nanosec = 0

                if topic == "/tf":
                    self.tf_buffer.set_transform(transform, "default_authority")
                elif topic == "/tf_static":
                    self.tf_buffer.set_transform_static(transform, "default_authority")

    def transform_pose(
        self, pose: Pose, target_frame: str, source_frame: str, time: Time
    ) -> Pose:
        """Transform a pose from source frame to target frame.

        Args:
            pose: The pose to transform
            target_frame: Target frame ID
            source_frame: Source frame ID
            time: Time to look up transform

        Returns:
            Transformed pose in target frame
        """
        transform = self.tf_buffer.lookup_transform(target_frame, source_frame, time)
        return tf2_geometry_msgs.do_transform_pose(pose, transform)


def add_distance_to_goal_metric(
    writer: ResimMetricsWriter, input_bag: Path, transform_manager: TransformManager
):
    """Add a metric showing distance to goal over time.

    Args:
        writer: Metrics writer instance
        input_bag: Path to the MCAP file
        transform_manager: Transform manager instance
    """
    # Track current position and goal
    current_odom: Odometry | None = None
    current_goal: PoseStamped | None = None
    transformed_goal: Pose | None = None
    goal_data: list[
        tuple[list[float], list[Time]]
    ] = []  # List of (distances, timestamps) for each goal
    current_distances: list[float] = []
    current_timestamps: list[Time] = []
    message_count = 0

    logger.info("Calculating distances to goal...")

    for topic, msg, timestamp in read_messages(
        str(input_bag), ["/chassis/odom", "/goal"]
    ):
        if topic == "/chassis/odom":
            current_odom = msg
        elif topic == "/goal":
            current_goal = msg
            # If we have data for the previous goal, save it
            if current_distances:
                goal_data.append((current_distances, current_timestamps))
                current_distances = []
                current_timestamps = []

            # Transform goal from map to odom frame when we get a new goal
            if current_goal is not None:  # Type check for current_goal
                try:
                    # Use the current odometry timestamp for the transform
                    if current_odom is not None:
                        timestamp = Time(
                            seconds=current_odom.header.stamp.sec,
                            nanoseconds=current_odom.header.stamp.nanosec,
                        )
                        transformed_goal = transform_manager.transform_pose(
                            current_goal.pose,
                            "odom",  # target frame (odom)
                            current_goal.header.frame_id,  # source frame (map)
                            timestamp,
                        )
                except Exception as e:
                    logger.warning("Could not transform goal", exc_info=e)
                    transformed_goal = None
                    continue

        # Calculate distance to goal if we have both pose and transformed goal
        if (
            current_odom is not None and transformed_goal is not None
        ):  # Type check for both
            try:
                # Calculate distance using odom coordinates
                dx = transformed_goal.position.x - current_odom.pose.pose.position.x
                dy = transformed_goal.position.y - current_odom.pose.pose.position.y
                distance = (dx * dx + dy * dy) ** 0.5

                # Only keep every 12th message (60Hz -> 5Hz)
                message_count += 1
                if message_count % 12 == 0:
                    current_distances.append(distance)
                    current_timestamps.append(
                        Time(
                            seconds=current_odom.header.stamp.sec,
                            nanoseconds=current_odom.header.stamp.nanosec,
                        )
                    )
                    logger.debug(f"Distance to goal: {distance:.2f}m")

            except Exception as e:
                logger.warning("Could not calculate distance", exc_info=e)
                continue

    # Add the last goal's data if we have any
    if current_distances:
        goal_data.append((current_distances, current_timestamps))

    if not goal_data:
        logger.warning("No distance data found")
        return

    # Create plot
    fig = go.Figure()

    # Define a sequence of colors that work well on dark backgrounds
    colors = RESIM_COLORS

    # Get the start time of the first goal for reference
    first_start_time = goal_data[0][1][0].nanoseconds

    # Add a line for each goal
    for i, (distances, timestamps) in enumerate(goal_data):
        # Convert timestamps to seconds from the start of the first goal
        times = [(t.nanoseconds - first_start_time) / 1e9 for t in timestamps]

        fig.add_trace(
            go.Scatter(
                x=times,
                y=distances,
                mode="lines",
                name=f"Goal {i + 1}",
                line=dict(color=colors[i % len(colors)], width=2),
            )
        )

        # Add metrics data for this goal
        add_metrics_data(writer, f"distance_to_goal_{i + 1}", times, distances)

    # Update layout
    fig.update_layout(
        xaxis=dict(title="Time (s)"),
        yaxis=dict(title="Distance to Goal (m)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(orientation="h", y=1.0, x=0.5, xanchor="center", yanchor="bottom"),
    )

    resim_plotly_style(fig)

    (
        writer.add_plotly_metric("Distance to Goal")
        .with_description(
            "Distance to each goal over time, measured from odometry. Each line represents a different goal."
        )
        .with_blocking(False)
        .with_plotly_data(str(fig.to_json()))
        .with_importance(MetricImportance.HIGH_IMPORTANCE)
        .with_status(MetricStatus.PASSED_METRIC_STATUS)
    )


def add_camera_gif_metric(
    writer: ResimMetricsWriter,
    input_bag: Path,
    output_path: Path,
    topic: str = "/front_stereo_camera/left/image_raw_throttled",
):
    """Create a GIF from camera images in an MCAP file, starting after the first goal.

    Args:
        writer: Metrics writer instance
        input_bag: Path to the MCAP file
        output_path: Path to save the output GIF
        topic: Camera topic to read images from
    """
    # First, find the timestamp of the first goal
    first_goal_time = None
    for _, msg, timestamp in read_messages(str(input_bag), ["/goal"]):
        first_goal_time = timestamp
        break

    if first_goal_time is None:
        logger.warning("No goal messages found")
        return

    # Read messages from the camera topic
    frames = []
    frame_count = 0

    logger.info("Processing frames...")

    for topic, msg, timestamp in read_messages(str(input_bag), [topic]):
        if timestamp < first_goal_time:
            continue

        if isinstance(msg, ROSImage):
            # Convert ROS Image to numpy array
            img = np.frombuffer(msg.data, dtype=np.uint8)
            img = img.reshape(msg.height, msg.width, -1)

            # Convert BGR to RGB if needed
            if msg.encoding == "bgr8":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize image to 25% of original size
            scale_percent = 25  # reduce to 25% of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

            frames.append(img)
            frame_count += 1

            if frame_count % 100 == 0:  # Log progress every 100 frames
                logger.info(f"Processed {frame_count} frames")

    if not frames:
        logger.warning("No valid frames found after goal")
        return

    # Save all frames as GIF
    logger.info("Saving final GIF...")
    imageio.mimsave(
        output_path,
        frames,
        fps=10,  # 2x speed (original 5Hz * 2)
        optimize=True,
        duration=0.1,
        quantizer="nq",
        loop=0,  # 0 means loop forever
    )

    logger.info(f"Saved optimized GIF to {output_path}")

    # Add GIF metric
    data = ExternalFileMetricsData(
        name="camera_gif_data", filename=str(output_path.name)
    )
    (
        writer.add_image_metric(name="Stereo Camera Feed")
        .with_description(
            "Camera feed from navigation, sped up 2x, starting after first goal received."
        )
        .with_blocking(False)
        .with_should_display(True)
        .with_status(MetricStatus.PASSED_METRIC_STATUS)
        .with_importance(MetricImportance.ZERO_IMPORTANCE)
        .with_image_data(data)
    )
    emit('camera_gif', {
        'filename': str(output_path.name)
    })


def radians_down_to_up_to_degrees(radians):
    """
    Convert radians to degrees. The bounds of the input are 0-2pi, and output is -180-180.
    The reference of the input is 0 degrees = South, the output is 0 degrees = North
    """
    corrected = (math.pi - radians) % (2 * math.pi)
    degrees = math.degrees(corrected)
    wrapped = ((degrees + 180) % 360) - 180
    return wrapped

def add_robot_trajectory_metric(
    writer: ResimMetricsWriter, input_bag: Path, transform_manager: TransformManager
):
    """Add a metric showing the robot's trajectory from a top-down view with goal points.

    Args:
        writer: Metrics writer instance
        input_bag: Path to the MCAP file
        transform_manager: Transform manager instance
    """
    # Track robot positions and goals
    x_positions = []
    y_positions = []
    timestamps: list[int] = []
    goal_x = []
    goal_y = []
    goal_angle = []
    message_count = 0

    logger.info("Calculating robot trajectory...")

    for topic, msg, timestamp in read_messages(
        str(input_bag), ["/chassis/odom", "/goal"]
    ):
        if topic == "/chassis/odom":
            assert isinstance(msg, Odometry)
            try:
                # Transform pose from odom to map frame
                transformed_pose = transform_manager.transform_pose(
                    msg.pose.pose,
                    "map",  # target frame
                    msg.header.frame_id,  # source frame (odom)
                    Time(
                        seconds=msg.header.stamp.sec,
                        nanoseconds=msg.header.stamp.nanosec,
                    ),
                )

                # Only keep every 12th message (60Hz -> 5Hz)
                message_count += 1
                if message_count % 12 == 0:
                    x_positions.append(transformed_pose.position.x)
                    y_positions.append(transformed_pose.position.y)
                    timestamps.append(timestamp)
            except Exception as e:
                logger.warning("Could not transform pose", exc_info=e)
                continue
        elif topic == "/goal":
            assert isinstance(msg, PoseStamped)
            goal_x.append(msg.pose.position.x)
            goal_y.append(msg.pose.position.y)
            so3 = SO3(quaternion=Quaternion(w=msg.pose.orientation.w, x=msg.pose.orientation.x, y=msg.pose.orientation.y, z=msg.pose.orientation.z))
            # NOTE: This won't handle non-zero x and y components of the goal orientation.
            goal_angle.append(radians_down_to_up_to_degrees(so3.log()[2]))

    if not x_positions:
        logger.warning("No odometry data found")
        return

    # Create plot
    df = pd.DataFrame(
        {
            "x": x_positions,
            "y": y_positions,
            "time": [
                (t - timestamps[0]) / 1e9 for t in timestamps
            ],  # Convert to seconds from start
        }
    )

    # Create the figure
    fig = go.Figure()

    # Add robot trajectory (swap axes: x on y-axis, y on x-axis)
    fig.add_trace(
        go.Scatter(
            x=df["y"],  # y position on x-axis
            y=df["x"],  # x position on y-axis
            mode="lines+markers",
            name="Robot Path",
            line=dict(color="#00ffff", width=3),
            marker=dict(
                size=6,
                color=df["time"].values,  # Convert Series to numpy array
                colorscale="Plasma",  # Better contrast on dark backgrounds
                showscale=True,
                colorbar=dict(title="Time (s)"),
            ),
        )
    )

    # Add goal points if any exist (swap axes)
    if goal_x and goal_y:
        fig.add_trace(
            go.Scatter(
                x=goal_y,  # y position on x-axis
                y=goal_x,  # x position on y-axis
                mode="markers",
                name="Goals",
                marker=dict(
                    size=12,
                    symbol="arrow",
                    color="yellow",
                    line=dict(color="black", width=1),
                    angle=goal_angle
                ),
            )
        )

    # Update layout (ensure axes are square and flip y-axis)
    fig.update_layout(
        xaxis=dict(
            title="Y Position (m)",
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(title="X Position (m)", autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(orientation="h", y=1.0, x=0.5, xanchor="center", yanchor="bottom"),
    )

    resim_plotly_style(fig)

    (
        writer.add_plotly_metric("Robot Trajectory")
        .with_description(
            "Top-down view of the robot's trajectory throughout the run, with goal points marked as yellow arrows."
        )
        .with_blocking(False)
        .with_plotly_data(str(fig.to_json()))
        .with_importance(MetricImportance.HIGH_IMPORTANCE)
        .with_status(MetricStatus.PASSED_METRIC_STATUS)
    )
    emit('robot_trajectory', {
        'raw_metric': str(fig.to_json())
    })

def add_time_to_goal_metric(writer: ResimMetricsWriter, input_bag: Path):
    first_goal_timestamp: float = math.inf
    end_timestamp: float = 0.0

    TIMEOUT_PATH = Path("/tmp/resim/inputs/logs/internal_timeout")
    maybe_timeout = None
    if TIMEOUT_PATH.is_file():
        with open(TIMEOUT_PATH, "r", encoding='utf-8') as fp:
            maybe_timeout = int(fp.read())

    msg: GoalStatus
    for _, msg, timestamp in read_messages(
        str(input_bag), ["/goal/status"]
    ):
        if msg.status == "NEW_GOAL":
            first_goal_timestamp = min(first_goal_timestamp, timestamp)
        elif msg.status == "COMPLETE":
            assert end_timestamp == 0.0, "/goal/status COMPLETE message seen more than once."
            end_timestamp = timestamp
    
    end_timestamp = (end_timestamp - first_goal_timestamp) / 1e9
    (
        writer.add_scalar_metric("Time to reach final goal")
        .with_description("Time between receiving first goal and reaching final goal. Fails if final goal not reached before timeout.")
        .with_status(MetricStatus.PASSED_METRIC_STATUS if maybe_timeout is None else MetricStatus.FAIL_BLOCK_METRIC_STATUS)
        .with_importance(MetricImportance.MEDIUM_IMPORTANCE)
        .with_value(end_timestamp if maybe_timeout is None else maybe_timeout)
        .with_unit("seconds")
    )
        


def add_metrics_data(
    writer: ResimMetricsWriter, name: str, times: Sequence[float], data: Sequence[float]
):
    timestamp_series = SeriesMetricsData(
        name=f"{name}_timestamps",
        series=seconds_array_to_timestamp(times),
        unit="seconds",
    )

    writer.add_metrics_data(
        SeriesMetricsData(
            name=name,
            series=np.array(data),
            index_data=timestamp_series,
            unit="metres",
        )
    )


def add_pose_difference_metric(
    writer: ResimMetricsWriter, input_bag: Path, transform_manager: TransformManager
):
    """Add a metric showing the difference between odometry and AMCL poses (at AMCL timestamps), comparing both in the map frame.

    Args:
        writer: Metrics writer instance
        input_bag: Path to the MCAP file
        transform_manager: Transform manager instance
    """

    logger.info(
        "Collecting odometry and AMCL messages in a single pass, transforming odom to map frame..."
    )

    timestamps = []
    position_diffs = []
    latest_odom = None

    # Single pass: messages are sorted by time
    for topic, msg, timestamp in read_messages(
        str(input_bag), ["/chassis/odom", "/amcl_pose"]
    ):
        if topic == "/chassis/odom":
            latest_odom = msg
        elif topic == "/amcl_pose":
            if latest_odom is not None:
                amcl_pos = msg.pose.pose.position
                # Transform odom pose to map frame at this timestamp
                try:
                    odom_pose = latest_odom.pose.pose
                    odom_header = latest_odom.header
                    odom_in_map = transform_manager.transform_pose(
                        odom_pose,
                        target_frame=msg.header.frame_id,
                        source_frame=odom_header.frame_id,
                        time=Time(
                            seconds=msg.header.stamp.sec,
                            nanoseconds=msg.header.stamp.nanosec,
                        ),
                    )
                    odom_pos = odom_in_map.position
                    pos_diff = np.sqrt(
                        (amcl_pos.x - odom_pos.x) ** 2
                        + (amcl_pos.y - odom_pos.y) ** 2
                        + (amcl_pos.z - odom_pos.z) ** 2
                    )
                    timestamps.append(timestamp)
                    position_diffs.append(pos_diff)
                except Exception as e:
                    logger.warning(
                        "Could not transform odom pose to map frame or compute difference",
                        exc_info=e,
                    )
                    continue

    if not timestamps:
        logger.warning("No pose difference data found")
        return

    # Create plot
    fig = go.Figure()
    # Convert timestamps to seconds from start
    times = [(t - timestamps[0]) / 1e9 for t in timestamps]

    add_metrics_data(writer, "pose_difference", times, position_diffs)

    # Add position difference trace
    fig.add_trace(
        go.Scatter(
            x=times,
            y=position_diffs,
            mode="markers+lines",
            name="Position Difference",
            line=dict(color=RESIM_TURQUOISE, width=2),
        )
    )
    # Update layout
    fig.update_layout(
        xaxis=dict(title="Time (s)"),
        yaxis=dict(title="Position Difference (m)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(orientation="h", y=1.0, x=0.5, xanchor="center", yanchor="bottom"),
    )
    resim_plotly_style(fig)
    (
        writer.add_plotly_metric("Localization Error")
        .with_description(
            "Difference in position between odometry and localization pose."
        )
        .with_blocking(False)
        .with_plotly_data(str(fig.to_json()))
        .with_importance(MetricImportance.HIGH_IMPORTANCE)
        .with_status(MetricStatus.PASSED_METRIC_STATUS)
    )


def run_experience_metrics(args):
    """Run the metrics for a single experience."""

    job_id = uuid.UUID(int=0)
    metrics_writer = ResimMetricsWriter(job_id=job_id)

    # Create transform manager and collect transforms
    transform_manager = TransformManager()
    transform_manager.collect_transforms(args.log_path)

    add_camera_gif_metric(
        writer=metrics_writer,
        input_bag=args.log_path,
        output_path=Path(args.output_path).with_name("camera.gif"),
    )

    # Add distance to goal metric
    add_distance_to_goal_metric(metrics_writer, args.log_path, transform_manager)

    # Add robot trajectory metric
    add_robot_trajectory_metric(metrics_writer, args.log_path, transform_manager)

    # Add pose difference metric
    add_pose_difference_metric(metrics_writer, args.log_path, transform_manager)
    add_time_to_goal_metric(metrics_writer, args.log_path)

    write_proto(metrics_writer, args.output_path)


def add_strip_or_box_plot(
    writer: ResimMetricsWriter,
    title: str,
    description: str,
    df: pd.DataFrame,
    value_col: str,
    value_label: str,
    group_col: str | None = None,
    group_label: str | None = None,
    hover_data: dict = {},
    color: str | None = None,
    color_discrete_map: dict | None = None,
):
    labels = {value_col: value_label}
    if group_label and group_col:
        labels[group_col] = group_label

    plot_args: dict[str, Any] = dict(
        labels=labels,
        hover_data=hover_data,
    )
    if color is not None:
        plot_args["color"] = color
    if color_discrete_map is not None:
        plot_args["color_discrete_map"] = color_discrete_map

    if len(df) > 60:
        fig = px.box(df, x=value_col, y=group_col, **plot_args)
    else:
        fig = px.strip(df, x=value_col, y=group_col, **plot_args)
    resim_plotly_style(fig)
    fig.update_layout({"xaxis": {"rangemode": "tozero"}, "showlegend": False})
    (
        writer.add_plotly_metric(title)
        .with_description(description)
        .with_importance(MetricImportance.MEDIUM_IMPORTANCE)
        .with_status(MetricStatus.PASSED_METRIC_STATUS)
        .with_plotly_data(str(fig.to_json()))
        .with_blocking(False)
    )


def extract_metric_series(job_to_metrics: dict, metric_name: str) -> dict:
    """Extract a metric series from job_to_metrics for all jobs."""
    return {
        str(pair[0]): cast(SeriesMetricsData, data).series
        for pair in job_to_metrics.items()
        if (
            data := next(
                (data for data in pair[1].metrics_data if data.name == metric_name),
                None,
            )
        )
        is not None
    }

def extract_scalar_metric_values(job_to_metrics: dict[uuid.UUID, UnpackedMetrics], metric_name: str) -> dict[str, float]:
    """Extract a scalar metric from job_to_metrics for all jobs."""
    values = {
        str(pair[0]): cast(ScalarMetric, data).value
        for pair in job_to_metrics.items()
        if (data := next(
            (data for data in pair[1].metrics if data.name == metric_name),
            None,
        ))
        is not None
    }
    values = {k: v for k,v in values.items() if v is not None}
    return values


def run_batch_metrics(args: argparse.Namespace) -> None:
    """Run metrics for a batch."""
    with open(
        args.batch_metrics_config_path, "r", encoding="utf-8"
    ) as metrics_config_file:
        metrics_config = json.load(metrics_config_file)

    token = metrics_config["authToken"]
    api_url = metrics_config["apiURL"]
    batch_id = metrics_config["batchID"]
    project_id = metrics_config["projectID"]
    job_to_metrics = fjm.fetch_job_metrics_by_batch(
        token=token,
        api_url=api_url,
        project_id=project_id,
        batch_id=uuid.UUID(batch_id),
    )

    # Create authenticated client
    client = AuthenticatedClient(
        base_url=metrics_config["apiURL"], token=metrics_config["authToken"]
    )

    # Get job metadata using the client
    batch_jobs_response = fetch_all_pages(
        list_jobs.sync,
        project_id=project_id,
        batch_id=batch_id,
        client=client,
        page_size=100,
    )

    # Convert job IDs to strings to match the format in metrics data
    job_id_to_experience_name = {
        str(job.job_id): job.experience_name
        for page in batch_jobs_response
        if page.jobs
        for job in page.jobs
    }

    writer = ResimMetricsWriter(uuid.uuid4())

    # Extract metrics data
    localization_error_per_job = extract_metric_series(
        job_to_metrics, "pose_difference"
    )

    # Create DataFrame for localization error statisticss
    localization_error_stats = pd.concat(
        [
            pd.DataFrame(
                {
                    "job_id": job_id,
                    "experience_name": job_id_to_experience_name[str(job_id)],
                    "max_error": np.max(localization_error_per_job[job_id]),
                },
                index=[0],  # Add index for single-row DataFrame
            )
            for job_id in localization_error_per_job.keys()
            if job_id in job_id_to_experience_name
        ],
        ignore_index=True,
    )

    # Plot maximum localization error
    add_strip_or_box_plot(
        writer,
        "Maximum Localization Error",
        "Maximum localization error (difference between odometry and AMCL pose) per job",
        localization_error_stats,
        value_col="max_error",
        value_label="meters",
        hover_data={"experience_name": True},
    )

    # Extract all distance_to_goal metrics (with prefix) and their timestamps
    distance_to_goal_per_job = {}
    timestamps_per_job = {}
    for job_id, metrics in job_to_metrics.items():
        job_id = str(job_id)
        for data in metrics.metrics_data:
            if data.name.startswith("distance_to_goal_") and not data.name.endswith(
                "_timestamps"
            ):
                if not isinstance(data, SeriesMetricsData):
                    continue

                if job_id not in distance_to_goal_per_job:
                    distance_to_goal_per_job[job_id] = []
                    timestamps_per_job[job_id] = []

                if data.index_data is None:
                    continue

                distance_to_goal_per_job[job_id].append(np.array(data.series))
                # index_data is a SeriesMetricsData with .series as np.ndarray of Timestamp objects
                timestamps_per_job[job_id].append(
                    np.array([ts.to_nanos() / 1e9 for ts in data.index_data.series])
                )

    # Calculate dwelling time for each goal in each job
    dwelling_times = []
    for job_id in distance_to_goal_per_job.keys():
        if job_id not in job_id_to_experience_name:
            continue

        for goal_idx, (distances, times) in enumerate(
            zip(distance_to_goal_per_job[job_id], timestamps_per_job[job_id])
        ):
            # Find goal transitions (when distance increases significantly)
            goal_transitions = np.where(np.diff(distances) > 0.5)[0] + 1
            goal_transitions = np.concatenate([[0], goal_transitions, [len(distances)]])

            # Calculate dwelling time for each goal segment
            for i in range(len(goal_transitions) - 1):
                goal_distances = distances[
                    goal_transitions[i] : goal_transitions[i + 1]
                ]
                goal_times = times[goal_transitions[i] : goal_transitions[i + 1]]

                # Calculate time spent within 0.5m
                close_mask = goal_distances < 0.5
                if np.any(close_mask):
                    # Find continuous segments where robot is within 0.5m
                    close_segments = np.where(
                        np.diff(np.concatenate([[False], close_mask, [False]]))
                    )[0].reshape(-1, 2)
                    # Sum up the time spent in each close segment
                    dwelling_time = sum(
                        goal_times[end - 1] - goal_times[start]
                        for start, end in close_segments
                    )
                    dwelling_times.append(
                        {
                            "job_id": job_id,
                            "experience_name": job_id_to_experience_name[job_id],
                            "goal_number": goal_idx + 1,
                            "dwelling_time": dwelling_time,
                        }
                    )

    if dwelling_times:
        # Create DataFrame with a unique index for each row
        dwelling_df = pd.DataFrame(dwelling_times, index=range(len(dwelling_times)))
        # Create color mapping for goals
        goal_colors = {i + 1: color for i, color in enumerate(RESIM_COLORS)}
        # Use add_strip_or_box_plot for dwelling time visualization
        add_strip_or_box_plot(
            writer,
            "Goal Dwelling Time",
            "Time spent within 0.5m of each goal for each job",
            dwelling_df,
            value_col="dwelling_time",
            value_label="seconds",
            hover_data={"goal_number": True, "experience_name": True},
            color="goal_number",
            color_discrete_map=goal_colors,
        )
    else:
        logger.warning("No dwelling times found to plot")

    # Calculate sum of all times to goal
    goal_sum = np.sum(list(extract_scalar_metric_values(job_to_metrics, "Time to reach final goal").values()))
    (
        writer.add_scalar_metric("Total time to reach final goal")
        .with_description("Sum of all 'Time to reach final goal' metrics across all experiences.")
        .with_status(MetricStatus.NOT_APPLICABLE_METRIC_STATUS)
        .with_importance(MetricImportance.MEDIUM_IMPORTANCE)
        .with_tag("RESIM_SUMMARY", "1")
        .with_value(goal_sum)
        .with_unit("seconds")
    )

    write_proto(writer, args.output_path)


def main():
    """Script entrypoint."""
    args = parse_args()
    if args.log_path.exists():
        logger.info(f"Running experience metrics for {str(args.log_path)}...")
        run_experience_metrics(args)
    elif args.batch_metrics_config_path.exists():
        logger.info(
            f"Running batch metrics for {str(args.batch_metrics_config_path)}..."
        )
        run_batch_metrics(args)
    else:
        logger.error("Couldn't find input files for experience or batch metrics jobs.")
        exit(1)


if __name__ == "__main__":
    main()
