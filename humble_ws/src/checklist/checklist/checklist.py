import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from pathlib import Path
from rclpy.task import Future


class Checklist(Node):
    def __init__(self, future):
        super().__init__('checklist')
        self.future = future
        self.subscription = self.create_subscription(
            TFMessage,
            '/tf',
            self.listener_callback,
            10
        )
        self.received_odom = False

        # Create a timer to log every 5 seconds
        self.timer = self.create_timer(5.0, self.timer_callback)
    
    def timer_callback(self):
        self.get_logger().info("Waiting on IsaacSim to be ready...")

    def listener_callback(self, msg: TFMessage):
        for t in msg.transforms:
            if t.header.frame_id == 'odom' and t.child_frame_id == 'base_link':
                self.received_odom = True
                self.get_logger().info('Received base_link -> odom')

        if self.received_odom:
            self.get_logger().info('Isaac is publishing on /tf... Starting Nav2')
            self.future.set_result(True)


def main(args=None):
    rclpy.init(args=args)
    future = Future()
    node = Checklist(future)
    rclpy.spin_until_future_complete(node, future)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
