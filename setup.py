from pathlib import Path

from setuptools import find_packages, setup


README = Path(__file__).parent / "README.md"

setup(
    name="openso-101",
    version="0.1.0",
    description="State-of-the-art open-source robot learning framework for the LeRobot SO-101 in Isaac Lab",
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires="==3.11.*",
    package_dir={"": "src"},
    packages=find_packages(where="src", include=["openso101*"]),
    include_package_data=True,
    package_data={
        "openso101.robots.so101": [
            "assets/usd/*",
            "assets/urdf/*",
            "assets/urdf/meshes/*",
            "LICENSE",
        ]
    },
    install_requires=[
        "isaaclab[all,isaacsim]==2.3.0",
        "torch==2.7.0",
        "torchvision==0.22.0",
        "numpy",
        "matplotlib",
        "tensorboard",
        "prettytable",
        "packaging",
    ],
    entry_points={
        "console_scripts": [
            "openso101=openso101.cli.main:main",
        ]
    },
)
