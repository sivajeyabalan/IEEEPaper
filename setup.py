"""
Setup script for GenIoT-Optimizer package.
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="geniot-optimizer",
    version="1.0.0",
    author="GenIoT-Optimizer Team",
    author_email="contact@geniot-optimizer.org",
    description="Generative AI for IoT Network Performance Simulation and Optimization",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/geniot-optimizer/geniot-optimizer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Networking",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
        "visualization": [
            "plotly>=5.15.0",
            "dash>=2.10.0",
            "streamlit>=1.25.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "geniot-demo=geniot_optimizer.examples.demo:main",
            "geniot-train=geniot_optimizer.training.pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "geniot_optimizer": [
            "configs/*.yaml",
            "data/*.csv",
            "models/*.pth",
        ],
    },
    keywords=[
        "iot", "generative-ai", "network-optimization", "deep-learning",
        "reinforcement-learning", "gan", "vae", "diffusion-models",
        "digital-twin", "smart-cities", "manufacturing", "smart-homes"
    ],
    project_urls={
        "Bug Reports": "https://github.com/geniot-optimizer/geniot-optimizer/issues",
        "Source": "https://github.com/geniot-optimizer/geniot-optimizer",
        "Documentation": "https://geniot-optimizer.readthedocs.io/",
        "Paper": "https://ieeexplore.ieee.org/document/xxxxx",
    },
)
