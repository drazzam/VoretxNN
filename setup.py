"""
Setup script for VortexNN package.
"""

from setuptools import setup, find_packages

setup(
    name="vortexnn",
    version="0.1.0",
    author="VortexNN Team",
    author_email="info@vortexnn.com",
    description="Advanced vortex-inspired graph neural networks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/vortexnn/vortexnn",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.7.0",
        "numpy>=1.19.0",
        "scikit-learn>=0.23.0",
        "matplotlib>=3.3.0",
    ],
)
