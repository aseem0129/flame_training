import numpy as np
from unittest.mock import mock_open, patch
from counterpart.export import Exporter

def test_export_glb_and_fbx():
    vertices = np.zeros((10, 3), dtype=np.float32)
    faces = np.zeros((20, 3), dtype=np.int32)
    texture = np.zeros((1024, 1024, 3), dtype=np.uint8)
    
    exporter = Exporter()
    
    # Test GLB export
    m = mock_open()
    with patch("builtins.open", m):
        exporter.export_glb(vertices, faces, texture, "out.glb")
        m.assert_called_once_with("out.glb", 'wb')
        handle = m()
        handle.write.assert_called_once_with(b'GLB_STUB')
    
    # Test FBX export
    m = mock_open()
    with patch("builtins.open", m):
        exporter.export_fbx(vertices, faces, texture, "out.fbx")
        m.assert_called_once_with("out.fbx", 'wb')
        handle = m()
        handle.write.assert_called_once_with(b'FBX_STUB') 