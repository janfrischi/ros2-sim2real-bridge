#!/usr/bin/env python3
"""
Script to verify ROS2 topics for BC policy runner
"""
# filepath: /home/pdzuser/franka_ros2_ws/src/franka_rl_bridge/franka_rl_bridge/verify_topics.py

import rclpy
from rclpy.node import Node
import time

class TopicVerifier(Node):
    def __init__(self):
        super().__init__('topic_verifier')
        
        # Required topics for BC policy runner
        self.required_topics = [
            '/joint_states',
            '/franka_robot_state_broadcaster/robot_state',
            '/franka_gripper/joint_states',
        ]
        
        # Optional topics
        self.optional_topics = [
            '/bc_policy/start',
            '/bc_policy/stop',
        ]
        
        # Output topics
        self.output_topics = [
            '/joint_position_controller/commands',
            '/gripper_position_controller/commands',
        ]
        
    def verify_topics(self):
        """Check if required topics are available"""
        print("Verifying ROS2 topics...")
        print("=" * 50)
        
        # Get all available topics
        topic_names_and_types = self.get_topic_names_and_types()
        available_topics = [name for name, _ in topic_names_and_types]
        
        print(f"Found {len(available_topics)} total topics")
        
        # Check required topics
        print("\nRequired Input Topics:")
        for topic in self.required_topics:
            if topic in available_topics:
                print(f"✓ {topic}")
            else:
                print(f"✗ {topic} - MISSING!")
        
        # Check output topics (these might not exist until controllers are running)
        print("\nOutput Topics (may not exist until controllers start):")
        for topic in self.output_topics:
            if topic in available_topics:
                print(f"✓ {topic}")
            else:
                print(f"? {topic} - Not found (controllers may not be running)")
        
        # Show Franka-related topics
        franka_topics = [t for t in available_topics if 'franka' in t.lower()]
        if franka_topics:
            print(f"\nAll Franka-related topics ({len(franka_topics)}):")
            for topic in sorted(franka_topics):
                print(f"  - {topic}")
        
        return True

def main():
    rclpy.init()
    
    verifier = TopicVerifier()
    verifier.verify_topics()
    
    rclpy.shutdown()

if __name__ == "__main__":
    main()