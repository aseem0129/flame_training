"""Face detection module using MediaPipe FaceMesh."""
from pathlib import Path
from typing import Optional
import cv2
import mediapipe as mp
import numpy as np

def run_facemesh(image_path: str) -> Optional[np.ndarray]:
    """Run MediaPipe FaceMesh on an image.
    
    Args:
        image_path: Path to input image
        
    Returns:
        Landmarks array of shape (468, 3) or None if no face detected
    """
    # Initialize MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
        
    # Convert to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process image
    results = mp_face_mesh.process(rgb_image)
    
    if not results.multi_face_landmarks:
        return None
        
    # Get landmarks
    landmarks = results.multi_face_landmarks[0].landmark
    
    # Convert to numpy array
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    
    return points 