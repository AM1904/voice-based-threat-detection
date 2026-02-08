from setuptools import setup, find_packages

setup(
    name="speech-to-text-module",
    version="1.0.0",
    description="Real-time speech-to-text for Voice-Based Threat Detection",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=["numpy>=1.24.0", "faster-whisper>=1.0.0"],
)
