"""Configuration models for Counterpart."""
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

class ProjectConfig(BaseModel):
    """Global project configuration."""
    project_root: Path = Field(default=Path.home() / ".counterpart")
    cache_dir: Path = Field(default=Path.home() / ".counterpart" / "cache")
    temp_dir: Path = Field(default=Path.home() / ".counterpart" / "tmp")
    
    def __init__(self, **data):
        super().__init__(**data)
        # Create directories if they don't exist
        self.project_root.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

class FaceDetectionConfig(BaseModel):
    """Configuration for face detection and cropping."""
    min_face_size: int = Field(default=100, description="Minimum face size in pixels")
    max_face_size: int = Field(default=1000, description="Maximum face size in pixels")
    face_mesh_model_path: Optional[Path] = Field(
        default=None,
        description="Path to MediaPipe FaceMesh .task file"
    )
    output_size: tuple[int, int] = Field(
        default=(512, 512),
        description="Size of cropped face images"
    ) 