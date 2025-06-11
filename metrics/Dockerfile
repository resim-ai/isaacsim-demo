FROM osrf/ros:humble-desktop

WORKDIR /app

RUN apt update && apt install -y python3-pip ros-humble-rosbag2-storage-mcap

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
