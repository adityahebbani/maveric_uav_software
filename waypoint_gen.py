#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import argparse

class ObstacleAvoidance:
    def __init__(self, connection_string, safety_margin=2.0):
        """
        Initialize the obstacle avoidance system
        
        Args:
            connection_string: Connection string for the drone
            safety_margin: Buffer distance (in meters) to maintain from obstacles
        """
        self.safety_margin = safety_margin
        # Connect to the drone
        print(f"Connecting to vehicle on: {connection_string}")
        self.vehicle = connect(connection_string, wait_ready=True)
        print("Connected to vehicle")
        
        # Store detected obstacles
        self.obstacles = []  # List of [x, y, z, width, height, depth] for each obstacle
        
        # Navigation parameters
        self.destination = None  # Final destination waypoint
        self.current_waypoints = []  # Generated waypoints for navigation
    
    def set_destination(self, lat, lon, alt):
        """Set the final destination waypoint"""
        self.destination = LocationGlobalRelative(lat, lon, alt)
        print(f"Destination set to: {lat}, {lon}, {alt}")
    
    def update_obstacles(self, detection_results):
        """
        Update obstacle list based on RF-DETR detection results
        
        Args:
            detection_results: Results from RF-DETR model containing object positions and dimensions
        """
        self.obstacles = []
        
        # Process detection results
        # This should be adapted to match the exact format of your RF-DETR output
        for detection in detection_results:
            # Extract object position (x, y, z) and dimensions (width, height, depth)
            # Example format - adjust based on your actual detection format
            obj_class = detection.get('class', 'unknown')
            confidence = detection.get('confidence', 0)
            
            # Only consider obstacles with high confidence
            if confidence > 0.5:
                x, y, z = detection.get('position', [0, 0, 0])
                w, h, d = detection.get('dimensions', [1, 1, 1])
                
                # Add obstacle to the list
                self.obstacles.append({
                    'position': [x, y, z],
                    'dimensions': [w, h, d],
                    'class': obj_class,
                    'confidence': confidence
                })
        
        print(f"Updated obstacles: {len(self.obstacles)} obstacles detected")
    
    def generate_waypoints(self):
        """
        Generate waypoints to navigate around obstacles to reach destination
        
        This is a simplified implementation - you'll need to enhance this with
        proper path planning algorithms like RRT, A*, or potential fields
        """
        if not self.destination:
            print("Error: Destination not set")
            return []
        
        # Get current position
        current_pos = self.vehicle.location.global_relative_frame
        
        # Clear current waypoints
        self.current_waypoints = []
        
        # If no obstacles, direct path to destination
        if not self.obstacles:
            self.current_waypoints = [self.destination]
            return self.current_waypoints
        
        # Simple obstacle avoidance strategy
        # This is where you would implement a more sophisticated algorithm
        # For now, we'll use a simple approach of going over obstacles
        
        # Add waypoint to climb above highest obstacle
        max_obstacle_height = 0
        for obstacle in self.obstacles:
            obstacle_top = obstacle['position'][2] + obstacle['dimensions'][1]/2
            max_obstacle_height = max(max_obstacle_height, obstacle_top)
        
        # Add safety margin
        safe_altitude = max_obstacle_height + self.safety_margin
        
        # Create intermediate waypoint at safe altitude
        intermediate_wp = LocationGlobalRelative(
            current_pos.lat, 
            current_pos.lon, 
            safe_altitude
        )
        self.current_waypoints.append(intermediate_wp)
        
        # Create intermediate waypoint at destination position but safe altitude
        intermediate_wp2 = LocationGlobalRelative(
            self.destination.lat, 
            self.destination.lon, 
            safe_altitude
        )
        self.current_waypoints.append(intermediate_wp2)
        
        # Add final destination
        self.current_waypoints.append(self.destination)
        
        print(f"Generated {len(self.current_waypoints)} waypoints")
        return self.current_waypoints
    
    def navigate_to_destination(self):
        """Navigate through generated waypoints to reach destination"""
        if not self.current_waypoints:
            print("No waypoints available. Generating new waypoints.")
            self.generate_waypoints()
        
        print("Starting navigation...")
        
        # Navigate through each waypoint
        for i, waypoint in enumerate(self.current_waypoints):
            print(f"Navigating to waypoint {i+1}/{len(self.current_waypoints)}")
            
            # Command the vehicle to move to the waypoint
            self.vehicle.simple_goto(waypoint)
            
            # Wait until reaching the waypoint (with timeout)
            timeout = 60  # seconds
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                # Get current location
                current = self.vehicle.location.global_relative_frame
                
                # Calculate distance to waypoint
                dist = self.get_distance_metres(current, waypoint)
                
                print(f"Distance to waypoint: {dist} meters")
                
                # Check if we've reached the waypoint (within 1 meter)
                if dist < 1.0:
                    print(f"Reached waypoint {i+1}")
                    break
                    
                time.sleep(1)
            
        print("Navigation complete!")
    
    def get_distance_metres(self, loc1, loc2):
        """
        Calculate distance between two LocationGlobalRelative objects
        """
        # Approximate radius of earth in meters
        R = 6371000
        
        lat1 = math.radians(loc1.lat)
        lon1 = math.radians(loc1.lon)
        lat2 = math.radians(loc2.lat)
        lon2 = math.radians(loc2.lon)
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        # Consider altitude difference as well
        altitude_diff = loc2.alt - loc1.alt
        
        # Use Euclidean distance in 3D
        return math.sqrt(distance**2 + altitude_diff**2)
    
    def close(self):
        """Close the connection to the vehicle"""
        self.vehicle.close()
        print("Connection to vehicle closed")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Drone obstacle avoidance navigation')
    parser.add_argument('--connect', 
                        help="Vehicle connection target string. If not specified, SITL automatically started.")
    parser.add_argument('--dest-lat', type=float, default=None,
                        help="Destination latitude")
    parser.add_argument('--dest-lon', type=float, default=None,
                        help="Destination longitude")
    parser.add_argument('--dest-alt', type=float, default=10,
                        help="Destination altitude (meters)")
    
    args = parser.parse_args()
    
    connection_string = args.connect or 'tcp:127.0.0.1:5760'  # Default to SITL
    
    # Initialize the obstacle avoidance system
    obstacle_avoidance = ObstacleAvoidance(connection_string)
    
    try:
        # Set destination if provided
        if args.dest_lat and args.dest_lon:
            obstacle_avoidance.set_destination(args.dest_lat, args.dest_lon, args.dest_alt)
        else:
            # Example destination
            obstacle_avoidance.set_destination(-35.363261, 149.165230, 20)
        
        # Example of updating obstacles from RF-DETR
        # In practice, this would be called when new detection results are available
        sample_detections = [
            {
                'class': 'person',
                'confidence': 0.95,
                'position': [10, 5, 0],
                'dimensions': [0.5, 1.8, 0.5]
            },
            {
                'class': 'tree',
                'confidence': 0.87,
                'position': [15, 8, 0],
                'dimensions': [3, 5, 3]
            }
        ]
        obstacle_avoidance.update_obstacles(sample_detections)
        
        # Generate waypoints and navigate
        obstacle_avoidance.generate_waypoints()
        obstacle_avoidance.navigate_to_destination()
        
    finally:
        # Close connection to the vehicle
        obstacle_avoidance.close()


if __name__ == '__main__':
    main()