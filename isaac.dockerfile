FROM osrf/ros:humble-desktop-full AS build

RUN apt update && \
	DEBIAN_FRONTEND=noninteractive apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator \
		python3-wstool build-essential python3-colcon-common-extensions
	
COPY humble_ws /humble_ws

WORKDIR /humble_ws
SHELL ["/bin/bash", "-c"]

RUN source /opt/ros/humble/setup.bash && \
    rosdep update && \
	rosdep install -i --from-path src --rosdistro humble -y && \
	colcon build

FROM nvcr.io/nvidia/isaac-sim:4.5.0 AS run

ENV OMNI_KIT_ALLOW_ROOT=1
ENV ACCEPT_EULA=Y

RUN apt update && apt install curl -y && \
	export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}') && \
	curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $UBUNTU_CODENAME)_all.deb" && \
	apt install /tmp/ros2-apt-source.deb && \
	apt update && \
	DEBIAN_FRONTEND=noninteractive apt install -y --allow-downgrades \
		libbrotli1=1.0.9-2build6 ros-humble-ros-base ros-humble-navigation2 \
		ros-humble-nav2-bringup python3-rosdep python3-rosinstall python3-rosinstall-generator \
		python3-wstool build-essential ros-humble-rviz2 ros-humble-rosbag2-storage-mcap python3-colcon-common-extensions

COPY --from=build /humble_ws /humble_ws

RUN rosdep init && \
    rosdep update && \
    rosdep install -i --from-path /humble_ws/src --rosdistro humble -y

COPY goals /goals
COPY entrypoint.sh /

ENTRYPOINT ["/entrypoint.sh"]
