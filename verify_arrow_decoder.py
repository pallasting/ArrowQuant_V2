import os
import time
import logging
from pathlib import Path
from llm_compression.tools.model_converter import ModelConverter, ConversionConfig
from llm_compression.inference.arrow_engine import ArrowEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DecoderTest")

def main():
    # Use ModelScope repo name for Qwen2.5 0.5B Instruct (Text Generation)
    model_id = "qwen/Qwen2.5-0.5B-Instruct" 
    output_dir = Path("./models/qwen2.5-0.5b-arrow")
    
    # Download from ModelScope
    logger.info(f"Downloading {model_id} from ModelScope...")
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        model_dir = snapshot_download(model_id)
        logger.info(f"Model downloaded to {model_dir}")
    except ImportError:
        logger.error("modelscope not installed. Run: pip install modelscope")
        return
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return
        
    # 1. Convert Model
    if not output_dir.exists() or not (output_dir / "weights.parquet").exists():
        logger.info(f"Converting model at {model_dir} to Arrow/Parquet format...")
        config = ConversionConfig(
            use_float16=True, # Critical for RAM efficiency and Intel HW acceleration
            compression="zstd",
            validate_output=False
        )
        converter = ModelConverter(config)
        result = converter.convert(model_dir, output_dir, model_type="causallm")
        
        if not result.success:
            logger.error(f"Conversion failed: {result.error_message}")
            return
        logger.info("Conversion successful!")
    else:
        logger.info(f"Model already converted at {output_dir}")
        
    # 2. Load Model via ArrowEngine (Zero-Copy)
    import gc
    gc.collect() 
    logger.info("Initializing ArrowEngine (Observe the Zero-Copy load speed!)...")
    start_time = time.time()
    
    # Enable Intel optimizations. ArrowEngine will automatically use oneDNN / MKL 
    # to accelerate PyTorch operations intrinsically.
    engine = ArrowEngine(
        str(output_dir), 
        device="cpu", 
        enable_intel_optimizations=True
    )
    load_time = time.time() - start_time
    logger.info(f"ArrowEngine LOADED in {load_time * 1000:.2f} ms!")
    
    # 3. Generation Engine Test
    # Using Instruct format for Qwen2.5-Instruct
    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the most important feature of an AI agent?<|im_end|>\n<|im_start|>assistant\n"
    logger.info("Testing with Instruct prompt...")
    
    logger.info("Starting generation loop (Autoregressive KV Cache mode)...")
    gen_start = time.time()
    
    # Requesting more tokens to see actual reasoning
    response = engine.generate(prompt, max_tokens=50, temperature=0.7)
    
    gen_time = time.time() - gen_start
    
    with open("generation_output.txt", "w", encoding="utf-8") as f:
        f.write("="*50 + "\n")
        f.write(f"PROMPT:\n{prompt}\n")
        f.write("-" * 50 + "\n")
        f.write(f"GENERATED RESPONSE:\n{response}\n")
        f.write("="*50 + "\n")
        
    logger.info("Full output saved to generation_output.txt")
    logger.info(f"Generation took {gen_time:.2f} seconds.")

if __name__ == "__main__":
    main()
