"""Landmark fusion module for combining face landmarks from multiple views."""
from pathlib import Path
from typing import List, Optional
import numpy as np

from .detect import run_facemesh

def fuse_landmarks(image_paths: List[str]) -> np.ndarray:
    """Fuse landmarks from multiple face images into a single set.
    
    Args:
        image_paths: List of paths to face images
        
    Returns:
        Fused landmarks array of shape (468, 3)
    """
    # Get landmarks for each image
    landmarks = []
    for path in image_paths:
        lm = run_facemesh(path)
        if lm is not None:
            landmarks.append(lm)
            
    if not landmarks:
        raise ValueError("No valid landmarks found in any image")
        
    # Stack landmarks and take mean
    stacked = np.stack(landmarks)
    fused = np.mean(stacked, axis=0)
    
    return fused 