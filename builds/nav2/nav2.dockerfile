FROM osrf/ros:humble-desktop AS build

RUN apt update && \
	DEBIAN_FRONTEND=noninteractive apt install -y ros-humble-rosbag2-storage-mcap python3-virtualenv python3-pip
	

WORKDIR /humble_ws
SHELL ["/bin/bash", "-c"]


# Layer optimization: Copy over all package xmls and run rosdep update since these change much less
# frequently than other code.

COPY humble_ws/src/isaacsim/package.xml ./src/isaacsim/package.xml
COPY humble_ws/src/checklist/package.xml ./src/checklist/package.xml
COPY humble_ws/src/isaac_ros2_messages/package.xml ./src/isaac_ros2_messages/package.xml
COPY humble_ws/src/metrics_emitter/package.xml ./src/metrics_emitter/package.xml
COPY humble_ws/src/navigation/carter_navigation/package.xml ./src/navigation/carter_navigation/package.xml

RUN source /opt/ros/humble/setup.bash && \
	apt update && \
    rosdep update && \
	rosdep install -i --from-path src --rosdistro humble -y

COPY /humble_ws/requirements.txt ./requirements.txt

RUN	python3 -m virtualenv venv && \
	source venv/bin/activate && \
	pip install --no-cache-dir -r requirements.txt

COPY humble_ws ./

# Rerun this in case we forgot any package xmls. Should be fast if not since we already installed
# all the deps above.
RUN source /opt/ros/humble/setup.bash && \
	apt update && \
    rosdep update && \
	rosdep install -i --from-path src --rosdistro humble -y
RUN source /opt/ros/humble/setup.bash && colcon build


COPY builds/nav2/goals /goals
COPY builds/nav2/entrypoint.sh /
COPY /.resim/metrics/config.yml ./resim_metrics_config.yml

ENTRYPOINT ["/entrypoint.sh"]
