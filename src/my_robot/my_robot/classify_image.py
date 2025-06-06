#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image as PILImage
import numpy as np
import os

class BirdClassifierNode(Node):
    """
    ROS2 node that subscribes to camera feed and performs real-time bird classification.
    Input: Camera feed from /depth_cam/rgb/image_raw topic
    Output: Displays camera feed with classification results
    """
    def __init__(self):
        super().__init__('bird_classifier_node')
        
        # Subscribe to the RGB camera topic
        self.cam_subscription = self.create_subscription(
            Image, 
            '/depth_cam/rgb/image_raw', 
            self.image_callback, 
            1
        )
        
        self.cv_bridge = CvBridge()
        self.latest_image = None
        
        # Get the package path
        self.package_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(self.package_path, 'models', 'best_bird_classifier.pth')
        
        # Load the PyTorch model
        self.get_logger().info('Loading PyTorch model...')
        try:
            self.model = self.load_model(model_path)
            self.get_logger().info('Model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {str(e)}')
            self.get_logger().error(f'Please ensure the model file exists at: {model_path}')
            raise
        
        # Define image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.get_logger().info('Bird classifier node initialized')
        
    def load_model(self, model_path):
        """
        Loads the trained PyTorch model from the specified path.
        
        Args:
            model_path (str): Path to the .pth model file
            
        Returns:
            torch.nn.Module: Loaded model in evaluation mode
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        # Initialize EfficientNet-B0 model from torchvision
        model = torchvision.models.efficientnet_b0(pretrained=False)
        num_classes = 3  # Model was trained for 3 classes
        
        # Modify the final layer to match your number of classes
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        
        # Load the trained weights
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def preprocess_image(self, cv_image):
        """
        Preprocesses the OpenCV image for model prediction.
        
        Args:
            cv_image (numpy.ndarray): OpenCV image in BGR format
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = PILImage.fromarray(rgb_image)
        
        # Apply transformations
        image_tensor = self.transform(pil_image)
        return image_tensor.unsqueeze(0)  # Add batch dimension

    def classify_image(self, image_tensor):
        """
        Classifies the input image using the loaded model.
        
        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor
            
        Returns:
            tuple: (predicted_class, confidence_score)
        """
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
            
            # Map class index to class name
            class_names = ['australian_magpie', 'king_parrot', 'little_corella']
            class_name = class_names[predicted_class]
            
        return class_name, confidence

    def image_callback(self, msg):
        """
        Callback function for processing incoming camera images.
        
        Args:
            msg (sensor_msgs.msg.Image): ROS2 Image message
        """
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_image = cv_image
            
            # Log image dimensions
            self.get_logger().info(f'Received image with dimensions: {cv_image.shape}')
            
            # Preprocess image
            image_tensor = self.preprocess_image(cv_image)
            
            # Classify image
            class_name, confidence = self.classify_image(image_tensor)
            
            # Draw results on the image
            result_text = f"{class_name}: {confidence:.2%}"
            cv2.putText(cv_image, result_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add a border around the image
            cv2.rectangle(cv_image, (0, 0), (cv_image.shape[1], cv_image.shape[0]), (0, 255, 0), 2)
            
            # Display the image
            self.get_logger().info('Attempting to display image window...')
            cv2.imshow("Bird Classification", cv_image)
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                self.get_logger().info('ESC key pressed, shutting down...')
                rclpy.shutdown()
            
        except Exception as e:
            self.get_logger().error(f'Error in image callback: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())

def main(args=None):
    rclpy.init(args=args)
    node = None
    
    try:
        # Create and run the node
        node = BirdClassifierNode()
        node.get_logger().info('Press Ctrl+C or ESC to exit.')
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        if node:
            node.get_logger().info('Keyboard interrupt detected, shutting down.')
    except Exception as e:
        if node:
            node.get_logger().error(f'Error running node: {str(e)}')
        else:
            print(f'Error initializing node: {str(e)}')
    finally:
        # Clean up
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 