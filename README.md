# Counterpart

A local-only prototype for generating FLAME avatars from selfie images.

## Features

- Face detection and cropping using MediaPipe FaceMesh
- FLAME model fitting and optimization
- Photometric texture baking
- FBX/GLB export with blendshapes
- OpenGL preview window

## Installation

1. Create a conda environment:
```bash
conda create -n counterpart python=3.11
conda activate counterpart
```

2. Install dependencies:
```bash
pip install -e .
```

3. Download required assets:
- FLAME model files from [flame.is.tue.mpg.de](https://flame.is.tue.mpg.de/)
- RingNet weights from [ringnet.is.tue.mpg.de](https://ringnet.is.tue.mpg.de/)
- MediaPipe FaceMesh model from [google.github.io/mediapipe](https://google.github.io/mediapipe/solutions/face_mesh)
- FrankMocap weights (optional) from [github.com/facebookresearch/frankmocap](https://github.com/facebookresearch/frankmocap)
- PIFuHD weights (optional) from [github.com/facebookresearch/pifuhd](https://github.com/facebookresearch/pifuhd)

## Usage

1. Process selfie images:
```bash
python -m counterpart.cli path/to/selfies/
```

2. Generate avatar:
```bash
python -m counterpart.generate path/to/cropped/faces/
```

3. Preview in viewer:
```bash
python -m counterpart.viewer path/to/avatar.fbx
```

## Development

- Type checking: `mypy src/`
- Tests: `pytest tests/`
- Linting: `ruff check src/`

## License

This project uses several third-party assets under various licenses:
- FLAME: CC-BY 4.0
- RingNet: MIT
- MediaPipe: Apache 2.0
- FrankMocap: BSD-3
- PIFuHD: BSD-3

See `licenses/` directory for full license texts. 