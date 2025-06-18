"""Tests for face detection functionality."""
import os
from pathlib import Path
import pytest
import numpy as np
import cv2

from counterpart.config import ProjectConfig, FaceDetectionConfig
from counterpart.face_detection import FaceDetector

@pytest.fixture
def config():
    """Create test configuration."""
    return FaceDetectionConfig(
        output_size=(256, 256),  # Smaller size for testing
        temp_dir=Path("test_output")
    )

@pytest.fixture
def detector(config):
    """Create face detector instance."""
    return FaceDetector(config)

def test_detect_face(detector):
    """Test face detection on a synthetic image."""
    # Create a synthetic face image
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    # Draw a simple face shape
    cv2.circle(image, (256, 256), 100, (255, 255, 255), -1)  # Face
    cv2.circle(image, (216, 216), 10, (0, 0, 0), -1)  # Left eye
    cv2.circle(image, (296, 216), 10, (0, 0, 0), -1)  # Right eye
    cv2.ellipse(image, (256, 296), (50, 30), 0, 0, 180, (0, 0, 0), 2)  # Smile
    
    # Detect face
    face = detector.detect_face(image)
    
    # Check if face was detected and cropped
    assert face is not None
    assert face.shape[:2] == (256, 256)  # Check output size

def test_process_image(detector, tmp_path):
    """Test processing a single image."""
    # Create test image
    image_path = tmp_path / "test.jpg"
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(image, (256, 256), 100, (255, 255, 255), -1)
    cv2.imwrite(str(image_path), image)
    
    # Process image
    output_path = detector.process_image(image_path)
    
    # Check output
    assert output_path is not None
    assert output_path.exists()
    assert output_path.suffix == ".png"
    
    # Check output image
    output_image = cv2.imread(str(output_path))
    assert output_image is not None
    assert output_image.shape[:2] == (256, 256)

def test_process_directory(detector, tmp_path):
    """Test processing a directory of images."""
    # Create test images
    for i in range(3):
        image_path = tmp_path / f"test_{i}.jpg"
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.circle(image, (256, 256), 100, (255, 255, 255), -1)
        cv2.imwrite(str(image_path), image)
    
    # Process directory
    output_paths = detector.process_directory(tmp_path)
    
    # Check outputs
    assert len(output_paths) == 3
    for path in output_paths:
        assert path.exists()
        assert path.suffix == ".png" 