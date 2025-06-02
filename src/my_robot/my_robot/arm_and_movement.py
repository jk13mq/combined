#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from servo_controller_msgs.msg import ServosPosition, ServoPosition
import time

class RobotController(Node):
    """
    A ROS2 node that combines arm and robot movement control with safety features.
    This class provides methods to control both the robotic arm and the robot's base movement
    with guaranteed return-to-safe-position after each movement.
    Input: Commands to move the arm and robot base
    Output: Publishes commands to both servo controller and cmd_vel topics
    """
    
    def __init__(self):
        super().__init__('robot_controller')
        
        # Create publishers for both arm and movement
        self.arm_publisher = self.create_publisher(ServosPosition, 'servo_controller', 10)
        self.movement_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Define positions
        self.home_position = {
            1: 500,  # Base joint
            2: 500,  # Shoulder joint
            3: 500,  # Elbow joint
            4: 500,  # Wrist joint
            5: 500,  # Wrist rotation
            6: 500   # Gripper
        }
        
        self.fold_position = {
            1: 500,  # Base rotation
            2: 250,  # Shoulder down
            3: 300,  # Elbow in
            4: 300,  # Wrist curled
            5: 500,  # Wrist rotation
            6: 200   # Gripper closed
        }
        
        self.raised_position = {
            1: 500,  # Base rotation
            2: 700,  # Shoulder up
            3: 700,  # Elbow up
            4: 500,  # Wrist straight
            5: 500,  # Wrist rotation
            6: 500   # Gripper open
        }
        
        self.get_logger().info('Robot controller initialized')

    def move_arm(self, joint_positions: dict, duration: float = 1.0):
        """
        Move multiple arm joints simultaneously
        
        Args:
            joint_positions (dict): Dictionary mapping joint IDs to their target positions
            duration (float): Time in seconds for the movement to complete
        """
        msg = ServosPosition()
        msg.position_unit = 'pulse'
        msg.duration = duration
        
        for joint_id, position in joint_positions.items():
            servo = ServoPosition()
            servo.id = joint_id
            servo.position = float(position)
            msg.position.append(servo)
            
        self.arm_publisher.publish(msg)
        self.get_logger().info(f"Moving arm joints: {joint_positions}")

    def safe_move_robot(self, linear_speed: float, angular_speed: float, duration: float):
        """
        Move the robot base and immediately return to original position
        Duration is automatically limited to 0.5 seconds maximum
        
        Args:
            linear_speed (float): Forward/backward speed in m/s
            angular_speed (float): Rotational speed in rad/s
            duration (float): Time to move in seconds (capped at 0.5s)
        """
        # Cap duration at 0.5 seconds
        duration = min(duration, 0.5)
        
        # Forward movement
        twist = Twist()
        twist.linear.x = linear_speed
        twist.angular.z = angular_speed
        
        end_time = time.time() + duration
        while time.time() < end_time and rclpy.ok():
            self.movement_publisher.publish(twist)
            time.sleep(0.1)
        
        # Immediate return movement
        twist.linear.x = -linear_speed
        twist.angular.z = -angular_speed
        
        end_time = time.time() + duration
        while time.time() < end_time and rclpy.ok():
            self.movement_publisher.publish(twist)
            time.sleep(0.1)
        
        # Stop the robot
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.movement_publisher.publish(twist)
        self.get_logger().info(f"Robot movement complete: linear={linear_speed}, angular={angular_speed}")

    def perform_complex_dance(self):
        """
        Perform a complex dance sequence with guaranteed safe movements.
        All wheel movements are limited to 0.5 seconds and return to original position.
        """
        try:
            # 1. Start with arm in home position
            self.get_logger().info("Starting dance sequence - Moving to home position")
            self.move_arm(self.home_position)
            time.sleep(1.0)

            # 2. Wave pattern with synchronized movement
            self.get_logger().info("Performing wave pattern with movement")
            for _ in range(3):
                # Forward wave
                self.safe_move_robot(0.2, 0.1, 0.4)
                self.move_arm({2: 700, 3: 600, 4: 500})
                time.sleep(0.3)
                
                # Backward wave
                self.safe_move_robot(-0.2, -0.1, 0.4)
                self.move_arm({2: 300, 3: 400, 4: 700})
                time.sleep(0.3)

            # 3. Spiral pattern
            self.get_logger().info("Performing spiral pattern")
            for i in range(4):
                angular_speed = 0.2 + (i * 0.05)
                self.safe_move_robot(0.15, angular_speed, 0.4)
                self.move_arm({1: 600, 2: 700, 3: 500})
                time.sleep(0.2)

            # 4. Zigzag with arm wave
            self.get_logger().info("Performing zigzag with arm wave")
            for _ in range(4):
                # Zig
                self.safe_move_robot(0.2, 0.3, 0.4)
                self.move_arm({1: 600, 2: 700, 3: 400})
                time.sleep(0.2)
                
                # Zag
                self.safe_move_robot(0.2, -0.3, 0.4)
                self.move_arm({1: 400, 2: 300, 3: 600})
                time.sleep(0.2)

            # 5. Wrist rotation pattern
            self.get_logger().info("Performing wrist rotation pattern")
            for _ in range(3):
                self.safe_move_robot(0.15, 0.2, 0.4)
                self.move_arm({4: 700, 5: 700})
                time.sleep(0.3)
                
                self.safe_move_robot(-0.15, -0.2, 0.4)
                self.move_arm({4: 300, 5: 300})
                time.sleep(0.3)

            # 6. Complex pattern combining all movements
            self.get_logger().info("Performing complex combined pattern")
            for _ in range(2):
                # Forward with raised arm
                self.safe_move_robot(0.2, 0.1, 0.4)
                self.move_arm(self.raised_position)
                time.sleep(0.3)
                
                # Backward with folded arm
                self.safe_move_robot(-0.2, -0.1, 0.4)
                self.move_arm(self.fold_position)
                time.sleep(0.3)
                
                # Rotate right with wave
                self.safe_move_robot(0.1, 0.4, 0.4)
                self.move_arm({2: 700, 3: 600, 4: 500})
                time.sleep(0.3)
                
                # Rotate left with wave
                self.safe_move_robot(0.1, -0.4, 0.4)
                self.move_arm({2: 300, 3: 400, 4: 700})
                time.sleep(0.3)

            # 7. Return to home position
            self.get_logger().info("Returning to home position")
            self.move_arm(self.home_position)
            time.sleep(1.0)

        except KeyboardInterrupt:
            self.get_logger().info("Dance interrupted by user")
            # Ensure we return to home position
            self.move_arm(self.home_position)
            self.safe_move_robot(0.0, 0.0, 0.1)  # Stop robot

def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    
    try:
        node.perform_complex_dance()
    except Exception as e:
        node.get_logger().error(f'Error during dance: {str(e)}')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()