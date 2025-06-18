from setuptools import setup, find_packages

setup(
    name="counterpart",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "mediapipe>=0.8.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pytorch3d>=0.7.0",
        "pillow>=9.0.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "counterpart=counterpart.cli:main",
        ],
    },
) 