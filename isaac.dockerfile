# Shaders image
FROM 909785973729.dkr.ecr.us-east-1.amazonaws.com/isaacsim-test-images:isaac-sim-4-5-0-shaders-8-7@sha256:073481cfda9a476c961bf9f5de4eb563a092545c0a12d3370a0ad9daf7b3b447 AS shaders

# Isaac Sim image
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

COPY --from=shaders /isaac-sim/kit/cache /isaac-sim/kit/cache
COPY /humble_ws /humble_ws

RUN rosdep init && \
	apt update && \
    rosdep update && \
    rosdep install -i --from-path /humble_ws/src --rosdistro humble -y && \
	cd /humble_ws && \
	colcon build

COPY goals /goals
COPY entrypoint.sh /

ENTRYPOINT ["/entrypoint.sh"]
