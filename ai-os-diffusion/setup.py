"""
AI-OS Unified Diffusion Architecture
Setup configuration
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="ai-os-diffusion",
    version="0.1.0",
    description="AI-OS Unified Diffusion Architecture with Rust Skeleton + Python Brain",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AI-OS Team",
    author_email="your.email@example.com",
    url="https://github.com/your-repo/ai-os-diffusion",
    packages=find_packages(exclude=["tests", "tests.*", "docs", "scripts"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pyarrow>=12.0.0",
        "transformers>=4.30.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "hypothesis>=6.82.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "tensorboard>=2.13.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Rust",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="diffusion, ai, machine-learning, rust, pytorch",
    project_urls={
        "Documentation": "https://github.com/your-repo/ai-os-diffusion/docs",
        "Source": "https://github.com/your-repo/ai-os-diffusion",
        "Tracker": "https://github.com/your-repo/ai-os-diffusion/issues",
    },
)
