import argparse
from enum import Enum
import logging
import math
import subprocess
import sys
from typing import Sequence
from pathlib import Path
import numpy as np
import rclpy.serialization
from resim.sdk.metrics import Emitter
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from foxglove_msgs.msg import CompressedVideo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Pose
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer
import tf2_geometry_msgs
from rclpy.time import Time, Duration

from resim.metrics.resim_style import resim_plotly_style
from resim.metrics.python.metrics_utils import Timestamp
from resim.transforms.python.quaternion import Quaternion
from resim.transforms.python.so3_python import SO3
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger("metrics.py")
logger.propagate = False
handler = logging.StreamHandler(stream=sys.stderr)
handler.setFormatter(logging.Formatter(fmt="%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)


DEFAULT_METRICS_CONFIG_PATH = "/app/resim_metrics_config.resim.yml"
DEFAULT_METRICS_PATH = "/tmp/resim/outputs/metrics.resim.jsonl"
DEFAULT_LOG_PATH = "/tmp/resim/inputs/logs/record_0.mcap"
DEFAULT_BATCH_METRICS_CONFIG_PATH = "/tmp/resim/inputs/batch_metrics_config.json"


class Topics(Enum):
    LOCAL_COSTMAP = "/local_costmap"
    LIDAR = "/hesai/pandar_points"


class FrameIds(Enum):
    BASE_LINK = "base_link"
    ODOM = "odom"
    MAP = "map"
    WORLD = "world"


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
        "--config-path",
        type=Path,
        default=DEFAULT_METRICS_CONFIG_PATH,
        help=f"Path to the metrics configuration file (default: {DEFAULT_METRICS_CONFIG_PATH})",
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

    # Create a dictionary mapping topic names to their types
    topic_type_map = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic not in topics:
            continue
        if topic not in topic_type_map:
            raise ValueError(f"topic {topic} not in bag")
        msg_type = get_message(topic_type_map[topic])
        msg = rclpy.serialization.deserialize_message(data, msg_type)
        yield topic, msg, timestamp


def seconds_array_to_timestamp(seconds: Sequence[float]) -> np.ndarray:
    return np.array(
        [Timestamp(secs=int(s), nanos=int((s - int(s)) * 1e9)) for s in seconds]
    )


def radians_down_to_up_to_degrees(radians):
    """
    Convert radians to degrees. The bounds of the input are 0-2pi, and output is -180-180.
    The reference of the input is 0 degrees = South, the output is 0 degrees = North
    """
    corrected = (math.pi - radians) % (2 * math.pi)
    degrees = math.degrees(corrected)
    wrapped = ((degrees + 180) % 360) - 180
    return wrapped


class MetricsRunner:
    """Runs experience metrics with shared state."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.namespace = self._identify_namespace()
        self.tf_buffer = Buffer(cache_time=Duration(seconds=pow(2, 31) - 1))
        self.seen_frame_pairs: set[tuple[str, str]] = set()
        self.collect_transforms(self.log_path)
        self.emitter: Emitter = Emitter(
            config_path=self.args.config_path,
            output_path=self.args.output_path,
        )

    @property
    def log_path(self) -> Path:
        return Path(self.args.log_path)

    @property
    def output_path(self) -> Path:
        return Path(self.args.output_path)

    @property
    def camera_output_path(self) -> Path:
        return self.output_path.with_name("camera.mp4")

    def _identify_namespace(self) -> str:
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=str(self.log_path), storage_id="mcap"),
            rosbag2_py.ConverterOptions(
                input_serialization_format="cdr", output_serialization_format="cdr"
            ),
        )
        topics = reader.get_all_topics_and_types()
        for topic in topics:
            if topic.name.endswith("/goal"):
                split_topic = topic.name.split("/")
                if len(split_topic) == 2:
                    return ""
                else:
                    return split_topic[1]

        raise ValueError("Did not find ~/goal/status topic in the bag.")

    def _namespaced_topic(self, topic: str) -> str:
        topic = topic.lstrip("/")
        if self.namespace:
            return f"/{self.namespace}/{topic}"
        return f"/{topic}"

    def collect_transforms(self, input_bag: Path):
        """Collect transforms from the bag file."""
        logger.info("Collecting transforms...")

        tf_topic = self._namespaced_topic("/tf")
        tf_static_topic = self._namespaced_topic("/tf_static")

        for topic, msg, _ in read_messages(str(input_bag), [tf_topic, tf_static_topic]):
            assert isinstance(msg, TFMessage)
            for transform in msg.transforms:
                if not (
                    transform.header.frame_id in [e.value for e in FrameIds]
                    and transform.child_frame_id in [e.value for e in FrameIds]
                ):
                    continue

                key = (transform.header.frame_id, transform.child_frame_id)
                if key not in self.seen_frame_pairs:
                    self.seen_frame_pairs.add(key)
                    transform.header.stamp.sec = 0
                    transform.header.stamp.nanosec = 0

                if topic == tf_topic:
                    self.tf_buffer.set_transform(transform, "default_authority")
                elif topic == tf_static_topic:
                    self.tf_buffer.set_transform_static(transform, "default_authority")

    def transform_pose(
        self, pose: Pose, target_frame: str, source_frame: str, time: Time
    ) -> Pose:
        """Transform a pose from source frame to target frame."""
        transform = self.tf_buffer.lookup_transform(target_frame, source_frame, time)
        return tf2_geometry_msgs.do_transform_pose(pose, transform)

    def run(self) -> None:
        """Run the metrics for a single experience."""
        self.emit_camera_video()
        self.emit_robot_trajectory_chart()
        self.emit_velocity_data()
        self.emit_nearest_human_distance_data()

    def emit_camera_video(self):
        """Extract MP4 from Foxglove compressed video messages after first goal."""
        camera_topic = self._namespaced_topic("/front_stereo_camera/left/image_raw/foxglove")
        first_goal_time = None
        goal_topic = self._namespaced_topic("/goal")
        for _, _, timestamp in read_messages(str(self.log_path), [goal_topic]):
            first_goal_time = timestamp
            break

        if first_goal_time is None:
            logger.warning("No goal messages found")
            return

        logger.info("Extracting compressed video stream after first goal...")
        raw_video_path = self.output_path.with_name("camera_stream.bin")
        frame_count = 0
        ffmpeg_input_format = None

        format_aliases = {
            "h264": "h264",
            "avc": "h264",
            "h.264": "h264",
            "h265": "hevc",
            "hevc": "hevc",
            "h.265": "hevc",
        }

        with raw_video_path.open("wb") as raw_video_file:
            for _, msg, timestamp in read_messages(str(self.log_path), [camera_topic]):
                if timestamp < first_goal_time:
                    continue

                if not isinstance(msg, CompressedVideo):
                    continue

                normalized_format = msg.format.strip().lower()
                if ffmpeg_input_format is None:
                    ffmpeg_input_format = format_aliases.get(normalized_format)
                    if ffmpeg_input_format is None:
                        logger.warning(
                            "Unsupported compressed video format '%s' on %s",
                            msg.format,
                            camera_topic,
                        )
                        if raw_video_path.exists():
                            raw_video_path.unlink()
                        return

                raw_video_file.write(bytes(msg.data))
                frame_count += 1

        if frame_count == 0:
            logger.warning("No compressed video frames found after first goal")
            if raw_video_path.exists():
                raw_video_path.unlink()
            return

        logger.info("Remuxing %d frames to MP4 with ffmpeg...", frame_count)
        command = [
            "ffmpeg",
            "-y",
            "-r",
            "25",
            "-f",
            ffmpeg_input_format,
            "-i",
            str(raw_video_path),
            "-vf",
            "scale=-2:720:flags=lanczos",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "24",
            "-maxrate",
            "12M",
            "-bufsize",
            "24M",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "25",
            "-movflags",
            "+faststart",
            str(self.camera_output_path),
        ]
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            if completed.stderr:
                logger.info(completed.stderr.strip())
        except subprocess.CalledProcessError as err:
            logger.error("ffmpeg failed while creating camera video: %s", err.stderr)
            return
        finally:
            if raw_video_path.exists():
                raw_video_path.unlink()

        logger.info("Saved camera video to %s", self.camera_output_path)
        self.emitter.emit(
            "camera_video",
            {
                "filename": str(self.camera_output_path.name),
                "camera_name": "front_stereo_camera_left",
            },
        )

    def emit_robot_trajectory_chart(self):
        """Add a metric showing the robot's trajectory from a top-down view with goal points."""
        x_positions = []
        y_positions = []
        timestamps: list[int] = []
        goal_x = []
        goal_y = []
        goal_angle = []
        message_count = 0

        logger.info("Calculating robot trajectory...")

        odom_topic = self._namespaced_topic("/chassis/odom")
        goal_topic = self._namespaced_topic("/goal")
        for topic, msg, timestamp in read_messages(
            str(self.log_path), [odom_topic, goal_topic]
        ):
            if topic == odom_topic:
                assert isinstance(msg, Odometry)
                try:
                    transformed_pose = self.transform_pose(
                        msg.pose.pose,
                        "map",
                        msg.header.frame_id,
                        Time(
                            seconds=msg.header.stamp.sec,
                            nanoseconds=msg.header.stamp.nanosec,
                        ),
                    )

                    message_count += 1
                    if message_count % 12 == 0:
                        x_positions.append(transformed_pose.position.x)
                        y_positions.append(transformed_pose.position.y)
                        timestamps.append(timestamp)
                except Exception as e:
                    logger.warning("Could not transform pose", exc_info=e)
                    continue
            elif topic == goal_topic:
                assert isinstance(msg, PoseStamped)
                goal_x.append(msg.pose.position.x)
                goal_y.append(msg.pose.position.y)
                so3 = SO3(
                    quaternion=Quaternion(
                        w=msg.pose.orientation.w,
                        x=msg.pose.orientation.x,
                        y=msg.pose.orientation.y,
                        z=msg.pose.orientation.z,
                    )
                )
                goal_angle.append(radians_down_to_up_to_degrees(so3.log()[2]))

        if not x_positions:
            logger.warning("No odometry data found")
            return

        df = pd.DataFrame(
            {
                "x": x_positions,
                "y": y_positions,
                "time": [(t - timestamps[0]) / 1e9 for t in timestamps],
            }
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["y"],
                y=df["x"],
                mode="lines+markers",
                name="Robot Path",
                line=dict(color="#00ffff", width=3),
                marker=dict(
                    size=6,
                    color=df["time"].values,
                    colorscale="Plasma",
                    showscale=True,
                    colorbar=dict(title="Time (s)"),
                ),
            )
        )

        if goal_x and goal_y:
            fig.add_trace(
                go.Scatter(
                    x=goal_y,
                    y=goal_x,
                    mode="markers",
                    name="Goals",
                    marker=dict(
                        size=12,
                        symbol="arrow",
                        color="yellow",
                        line=dict(color="black", width=1),
                        angle=goal_angle,
                    ),
                )
            )

        fig.update_layout(
            xaxis=dict(title="Y Position (m)", scaleanchor="y", scaleratio=1),
            yaxis=dict(title="X Position (m)", autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(
                orientation="h", y=1.0, x=0.5, xanchor="center", yanchor="bottom"
            ),
        )

        resim_plotly_style(fig)

        self.emitter.emit("robot_trajectory", {"raw_metric": str(fig.to_json())})

    def emit_velocity_data(self):
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=str(self.log_path), storage_id="mcap"),
            rosbag2_py.ConverterOptions(
                input_serialization_format="cdr", output_serialization_format="cdr"
            ),
        )

        topic_type_map = {
            topic.name: topic.type for topic in reader.get_all_topics_and_types()
        }
        odom_topic = self._namespaced_topic("/chassis/odom")
        if odom_topic not in topic_type_map:
            raise ValueError(f"topic {odom_topic} not in bag")

        msg_type = get_message(topic_type_map[odom_topic])
        last_emit_time = None
        emit_interval_ns = 100_000_000

        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            if topic == odom_topic and (
                last_emit_time is None or (timestamp - last_emit_time) >= emit_interval_ns
            ):
                msg = rclpy.serialization.deserialize_message(data, msg_type)
                assert isinstance(msg, Odometry)
                self.emitter.emit(
                    "odom_linear_velocity",
                    {
                        "x": msg.twist.twist.linear.x,
                        "y": msg.twist.twist.linear.y,
                        "z": msg.twist.twist.linear.z,
                    },
                    timestamp=timestamp,
                )
                last_emit_time = timestamp

    def emit_nearest_human_distance_data(self):
        """Emit nearest-human distance from odometry and ground-truth pedestrian poses."""
        logger.info("Starting nearest-human distance emission...")
        odom_topic = self._namespaced_topic("/chassis/odom")
        ped_topics = get_topics_by_prefix(self.log_path, "/ground_truth/ped/")
        if not ped_topics:
            logger.warning("No pedestrian ground-truth topics found")
            return

        latest_ped_poses: dict[str, PoseStamped] = {}
        last_emit_time = None
        emit_interval_ns = 100_000_000
        topics = [odom_topic, *ped_topics]
        total_odom_samples = 0
        ped_pose_updates = 0
        transform_failures = 0
        emitted_samples = 0

        for topic, msg, timestamp in read_messages(str(self.log_path), topics):
            if topic == odom_topic:
                assert isinstance(msg, Odometry)
                total_odom_samples += 1
                if last_emit_time is not None and (timestamp - last_emit_time) < emit_interval_ns:
                    continue
                if not latest_ped_poses:
                    continue

                odom_time = Time(
                    seconds=msg.header.stamp.sec,
                    nanoseconds=msg.header.stamp.nanosec,
                )
                robot_x = msg.pose.pose.position.x
                robot_y = msg.pose.pose.position.y

                nearest_distance = math.inf
                nearest_person_name = None
                for ped_topic, ped_pose_stamped in latest_ped_poses.items():
                    source_frame = ped_pose_stamped.header.frame_id
                    if not source_frame:
                        continue

                    try:
                        if source_frame == "odom":
                            ped_pose = ped_pose_stamped.pose
                        else:
                            candidate_source_frames = [source_frame]
                            if source_frame == "world":
                                candidate_source_frames.append("map")

                            transformed_pose = None
                            for candidate_source_frame in candidate_source_frames:
                                try:
                                    transformed_pose = self.transform_pose(
                                        ped_pose_stamped.pose,
                                        "odom",
                                        candidate_source_frame,
                                        odom_time,
                                    )
                                    break
                                except Exception:
                                    try:
                                        transformed_pose = self.transform_pose(
                                            ped_pose_stamped.pose,
                                            "odom",
                                            candidate_source_frame,
                                            Time(),
                                        )
                                        break
                                    except Exception:
                                        continue
                            if transformed_pose is None:
                                raise RuntimeError(
                                    f"Could not transform ped pose from any source frame candidate: {candidate_source_frames}"
                                )
                            ped_pose = transformed_pose
                    except Exception as e:
                        transform_failures += 1
                        logger.debug("Could not transform pedestrian pose", exc_info=e)
                        continue

                    dx = ped_pose.position.x - robot_x
                    dy = ped_pose.position.y - robot_y
                    distance_m = math.hypot(dx, dy)
                    if distance_m < nearest_distance:
                        nearest_distance = distance_m
                        nearest_person_name = ped_topic.rsplit("/", maxsplit=1)[-1]

                if nearest_person_name is None or not math.isfinite(nearest_distance):
                    continue

                self.emitter.emit(
                    "nearest_human_distance",
                    {
                        "distance_m": nearest_distance,
                        "person_name": nearest_person_name,
                    },
                    timestamp=timestamp,
                )
                emitted_samples += 1
                last_emit_time = timestamp
            else:
                assert isinstance(msg, PoseStamped)
                latest_ped_poses[topic] = msg
                ped_pose_updates += 1

        logger.info(
            "Nearest-human emission complete: %d emitted samples, %d ped updates, %d odom samples, %d transform failures, %d pedestrian topics",
            emitted_samples,
            ped_pose_updates,
            total_odom_samples,
            transform_failures,
            len(ped_topics),
        )


def get_topics_by_prefix(input_bag: Path, prefix: str) -> list[str]:
    """Return all topic names in a bag that match a prefix."""
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(input_bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    return sorted(
        topic.name
        for topic in reader.get_all_topics_and_types()
        if topic.name.startswith(prefix)
    )
    

def main():
    """Script entrypoint."""
    args = parse_args()
    if args.log_path.exists():
        logger.info(f"Running experience metrics for {str(args.log_path)}...")
        MetricsRunner(args).run()
    elif args.batch_metrics_config_path.exists():
        logger.info(
            f"Skipping batch metrics for {str(args.batch_metrics_config_path)}..."
        )
    else:
        logger.error("Couldn't find input files for experience or batch metrics jobs.")
        exit(1)


if __name__ == "__main__":
    main()
