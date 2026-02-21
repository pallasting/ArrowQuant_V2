from setuptools import setup, find_packages

setup(
    name="llm_compression",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ollama",
        "sentence-transformers",
        "numpy",
        "networkx",
        "aiohttp",
        "torch",
        "transformers",
        "zstandard",
        "pyarrow",
        "pyyaml",
        "fastapi",
    ],
    extras_require={
        "angelslim": [
            "angelslim>=0.3.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
            "hypothesis>=6.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "all": [
            "angelslim>=0.3.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
            "hypothesis>=6.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
)
