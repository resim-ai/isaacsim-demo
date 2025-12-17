# Copyright 2025 ReSim, Inc.
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import re

import rclpy
from rclpy.node import Node
from simulation_interfaces.srv import GetEntities, DeleteEntity
from simulation_interfaces.msg import EntityFilters, Result


class ObstacleCleanup(Node):
    def __init__(self):
        super().__init__("obstacle_cleanup")
        
        # Create service clients
        self.get_entities_client = self.create_client(GetEntities, "/isaacsim/GetEntities")
        self.delete_entity_client = self.create_client(DeleteEntity, "/isaacsim/DeleteEntity")
        
        # Wait for services to be available
        self.get_logger().info("Waiting for Isaac Sim services...")
        if not self.get_entities_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("GetEntities service not available")
            return
        
        if not self.delete_entity_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("DeleteEntity service not available")
            return
        
        self.get_logger().info("Services available, starting cleanup...")
        self.cleanup_obstacles()
    
    def cleanup_obstacles(self):
        """Find and delete all top-level entities created by obstacle_generator."""
        # Create filter to find all Obstacle entities
        # Using regex pattern to match Obstacle1, Obstacle2, Obstacle3, etc.
        filters = EntityFilters()
        filters.filter = "Obstacle[0-9]+"  # POSIX Extended regex pattern
        
        # Call GetEntities service
        request = GetEntities.Request()
        request.filters = filters
        
        self.get_logger().info("Querying for Obstacle entities...")
        future = self.get_entities_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        response = future.result()
        if response is None:
            self.get_logger().error("Failed to get entities: service call failed")
            return
        
        if response.result.result != Result.RESULT_OK:
            self.get_logger().error(f"Failed to get entities: {response.result.error_message}")
            return
        
        # Filter to only include top-level entities (Obstacle1, Obstacle2, Obstacle3)
        # Match patterns like /Pallet1, /Pallet2, /Pallet3
        top_level_pattern = re.compile(r'^/Obstacle[0-9]+$')
        entities = [e for e in response.entities if top_level_pattern.match(e)]
        
        self.get_logger().info(f"Found {len(entities)} top-level Obstacle entities to delete")
        
        if len(entities) == 0:
            self.get_logger().info("No Obstacle entities found. Nothing to clean up.")
            return
        
        # Delete each entity
        deleted_count = 0
        failed_count = 0
        
        for entity in entities:
            self.get_logger().info(f"Deleting entity: {entity}")
            delete_request = DeleteEntity.Request()
            delete_request.entity = entity
            
            delete_future = self.delete_entity_client.call_async(delete_request)
            rclpy.spin_until_future_complete(self, delete_future)
            
            delete_response = delete_future.result()
            if delete_response is None:
                self.get_logger().error(f"Failed to delete {entity}: service call failed")
                failed_count += 1
                continue
            
            if delete_response.result.result == Result.RESULT_OK:
                self.get_logger().info(f"Successfully deleted {entity}")
                deleted_count += 1
            else:
                self.get_logger().warn(f"Failed to delete {entity}: {delete_response.result.error_message}")
                failed_count += 1
        
        self.get_logger().info(f"Cleanup complete: {deleted_count} deleted, {failed_count} failed")


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleCleanup()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

