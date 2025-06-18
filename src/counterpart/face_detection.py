"""Face detection and cropping using MediaPipe FaceMesh."""
from pathlib import Path
from typing import Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

from .config import FaceDetectionConfig

class FaceDetector:
    """Detects and crops faces using MediaPipe FaceMesh."""
    
    def __init__(self, config: FaceDetectionConfig):
        """Initialize the face detector.
        
        Args:
            config: Face detection configuration
        """
        self.config = config
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def detect_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect and crop face from image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Cropped face image or None if no face detected
        """
        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
            
        # Get face landmarks
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Convert landmarks to pixel coordinates
        h, w = image.shape[:2]
        points = np.array([[int(lm.x * w), int(lm.y * h)] for lm in landmarks])
        
        # Get face bounding box with padding
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        
        # Add padding
        padding = int((x_max - x_min) * 0.1)
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Crop face
        face = image[y_min:y_max, x_min:x_max]
        
        # Resize to target size
        face = cv2.resize(face, self.config.output_size)
        
        return face
        
    def process_image(self, image_path: Path) -> Optional[Path]:
        """Process a single image and save cropped face.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Path to saved cropped face or None if no face detected
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        # Detect and crop face
        face = self.detect_face(image)
        if face is None:
            return None
            
        # Save cropped face
        output_path = self.config.temp_dir / f"face_{image_path.stem}.png"
        cv2.imwrite(str(output_path), face)
        
        return output_path
        
    def process_directory(self, input_dir: Path) -> list[Path]:
        """Process all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            
        Returns:
            List of paths to cropped faces
        """
        output_paths = []
        for image_path in input_dir.glob("*.{jpg,jpeg,png}"):
            try:
                output_path = self.process_image(image_path)
                if output_path is not None:
                    output_paths.append(output_path)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                
        return output_paths 