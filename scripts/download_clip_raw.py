
"""
Raw downloader for CLIP model files.
Downloads openai/clip-vit-base-patch32 from HuggingFace to local disk.
Uses standard library to avoid dependency issues.
"""

import os
import urllib.request
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_ID = "openai/clip-vit-base-patch32"
BASE_URL = f"https://huggingface.co/{MODEL_ID}/resolve/main"
FILES_TO_DOWNLOAD = [
    "config.json",
    "pytorch_model.bin",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json"
]
DOWNLOAD_DIR = Path("models/raw/clip-vit-base-patch32")

def download_file(filename: str):
    url = f"{BASE_URL}/{filename}"
    output_path = DOWNLOAD_DIR / filename

    if output_path.exists():
        logger.info(f"Skipping {filename} (already exists)")
        return

    logger.info(f"Downloading {filename} from {url}...")
    try:
        # User-Agent is often required by HF
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req) as response:
            content = response.read()
            with open(output_path, 'wb') as f:
                f.write(content)
        logger.info(f"Successfully downloaded {filename}")
    except Exception as e:
        logger.error(f"Failed to download {filename}: {e}")
        # Try finding a mirror if needed, or handle safetensors
        if filename == "pytorch_model.bin":
            logger.info("Attempting to download model.safetensors instead...")
            try:
                alt_filename = "model.safetensors"
                url = f"{BASE_URL}/{alt_filename}"
                output_path = DOWNLOAD_DIR / alt_filename
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    content = response.read()
                    with open(output_path, 'wb') as f:
                        f.write(content)
                logger.info(f"Successfully downloaded {alt_filename}")
            except Exception as e2:
                logger.error(f"Failed to download model.safetensors: {e2}")

def main():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {MODEL_ID} to {DOWNLOAD_DIR}")

    for filename in FILES_TO_DOWNLOAD:
        download_file(filename)

    logger.info("Download process completed.")

if __name__ == "__main__":
    main()
