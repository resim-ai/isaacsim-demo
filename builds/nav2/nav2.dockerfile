FROM osrf/ros:humble-desktop AS build

RUN apt update && \
	DEBIAN_FRONTEND=noninteractive apt install -y ros-humble-rosbag2-storage-mcap
	

WORKDIR /humble_ws
SHELL ["/bin/bash", "-c"]


# Layer optimization: Copy over all package xmls and run rosdep update since these change much less
# frequently than other code.

COPY humble_ws/src/isaacsim/package.xml ./src/isaacsim/package.xml
COPY humble_ws/src/isaac_tutorials/package.xml ./src/isaac_tutorials/package.xml
COPY humble_ws/src/custom_message/package.xml ./src/custom_message/package.xml
COPY humble_ws/src/ackermann_control/cmdvel_to_ackermann/package.xml ./src/ackermann_control/cmdvel_to_ackermann/package.xml
COPY humble_ws/src/moveit/isaac_moveit/package.xml ./src/moveit/isaac_moveit/package.xml
COPY humble_ws/src/checklist/package.xml ./src/checklist/package.xml
COPY humble_ws/src/humanoid_locomotion_policy_example/h1_fullbody_controller/package.xml ./src/humanoid_locomotion_policy_example/h1_fullbody_controller/package.xml
COPY humble_ws/src/isaac_ros2_messages/package.xml ./src/isaac_ros2_messages/package.xml
COPY humble_ws/src/navigation/carter_navigation/package.xml ./src/navigation/carter_navigation/package.xml
COPY humble_ws/src/navigation/iw_hub_navigation/package.xml ./src/navigation/iw_hub_navigation/package.xml
COPY humble_ws/src/navigation/isaac_ros_navigation_goal/package.xml ./src/navigation/isaac_ros_navigation_goal/package.xml

RUN source /opt/ros/humble/setup.bash && \
	apt update && \
    rosdep update && \
	rosdep install -i --from-path src --rosdistro humble -y


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

ENTRYPOINT ["/entrypoint.sh"]
