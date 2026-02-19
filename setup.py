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
)
