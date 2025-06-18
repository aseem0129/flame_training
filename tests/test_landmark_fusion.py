"""Tests for landmark fusion functionality."""
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from counterpart.landmark import fuse_landmarks

# Load fake landmarks
FAKE_LM = np.load("tests/fixtures/face_landmarks.npy")

DUMMY_IMAGE = np.zeros((256, 256, 3), dtype=np.uint8)

class MockFaceMesh:
    def __init__(self, *args, **kwargs):
        pass
        
    def process(self, image):
        result = MagicMock()
        result.multi_face_landmarks = [MagicMock()]
        result.multi_face_landmarks[0].landmark = [
            MagicMock(x=0.0, y=0.0, z=0.0) for _ in range(468)
        ]
        return result

def test_fuse_landmarks():
    """Test landmark fusion with mocked MediaPipe."""
    with patch("mediapipe.solutions.face_mesh.FaceMesh", MockFaceMesh), \
         patch("cv2.imread", return_value=DUMMY_IMAGE):
        fused = fuse_landmarks(["img1.jpg", "img2.jpg", "img3.jpg"])
        assert fused.shape == (468, 3)
        
def test_fuse_landmarks_no_faces():
    """Test landmark fusion when no faces are detected."""
    class MockFaceMeshNoFaces:
        def __init__(self, *args, **kwargs):
            pass
            
        def process(self, image):
            result = MagicMock()
            result.multi_face_landmarks = None
            return result
            
    with patch("mediapipe.solutions.face_mesh.FaceMesh", MockFaceMeshNoFaces), \
         patch("cv2.imread", return_value=DUMMY_IMAGE):
        with pytest.raises(ValueError, match="No valid landmarks found"):
            fuse_landmarks(["img1.jpg", "img2.jpg", "img3.jpg"])
            
def test_fuse_landmarks_mixed():
    """Test landmark fusion with some successful and some failed detections."""
    class MockFaceMeshMixed:
        def __init__(self, *args, **kwargs):
            self.counter = 0
            
        def process(self, image):
            result = MagicMock()
            if self.counter % 2 == 0:
                result.multi_face_landmarks = [MagicMock()]
                result.multi_face_landmarks[0].landmark = [
                    MagicMock(x=0.0, y=0.0, z=0.0) for _ in range(468)
                ]
            else:
                result.multi_face_landmarks = None
            self.counter += 1
            return result
            
    with patch("mediapipe.solutions.face_mesh.FaceMesh", MockFaceMeshMixed), \
         patch("cv2.imread", return_value=DUMMY_IMAGE):
        fused = fuse_landmarks(["img1.jpg", "img2.jpg", "img3.jpg"])
        assert fused.shape == (468, 3) 