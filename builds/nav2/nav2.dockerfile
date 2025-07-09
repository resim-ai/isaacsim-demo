FROM osrf/ros:humble-desktop AS build

RUN apt update && \
	DEBIAN_FRONTEND=noninteractive apt install -y ros-humble-rosbag2-storage-mcap
	
COPY humble_ws /humble_ws

WORKDIR /humble_ws
SHELL ["/bin/bash", "-c"]

RUN source /opt/ros/humble/setup.bash && \
	apt update && \
    rosdep update && \
	rosdep install -i --from-path src --rosdistro humble -y && \
	colcon build

COPY builds/nav2/goals /goals
COPY builds/nav2/entrypoint.sh /

ENTRYPOINT ["/entrypoint.sh"]
