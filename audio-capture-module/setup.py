"""
Audio Capture Module - Setup Script

Install with:
    pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="audio-capture-module",
    version="1.0.0",
    description="Real-time audio capture module for Voice-Based Threat Detection System",
    author="Voice Threat Detection Team",
    packages=find_packages(),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "sounddevice>=0.4.6",
        "numpy>=1.24.0",
    ],
    extras_require={
        "viz": ["matplotlib>=3.7.0"],
        "test": ["psutil>=5.9.0"],
        "all": ["matplotlib>=3.7.0", "psutil>=5.9.0"],
    },
    entry_points={
        "console_scripts": [
            "audio-capture-demo=demo:main",
        ],
    },
)
