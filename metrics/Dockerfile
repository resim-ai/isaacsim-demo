FROM osrf/ros:humble-desktop

WORKDIR /app

RUN apt update && apt install -y python3-pip ros-humble-rosbag2-storage-mcap

SHELL ["/bin/bash", "-c"]

COPY humble_ws/src/custom_message /humble_ws/src/custom_message
RUN source /opt/ros/humble/setup.bash && \
    cd /humble_ws && \
    colcon build

COPY metrics/requirements.txt .
RUN pip install -r requirements.txt

COPY metrics /app
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
