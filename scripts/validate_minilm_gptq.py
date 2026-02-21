import torch
from llm_compression.inference.precision_validator import PrecisionValidator

def main():
    original_path = "models/minilm/weights.parquet"
    quantized_path = "models/minilm/weights_int2_gptq.parquet"
    
    test_texts = [
        "Self-evolving memory systems are the future of AI operating systems.",
        "The quick brown fox jumps over the lazy dog.",
        "Precision and recall are key metrics in information retrieval.",
        "Multimodal fusion allows integration of vision and text data.",
        "Quantization reduces the memory footprint of large language models."
    ]
    
    validator = PrecisionValidator(cosine_threshold=0.88)
    result = validator.validate(
        original_model_path=original_path,
        quantized_model_path=quantized_path,
        test_texts=test_texts
    )
    
    print("\n" + "="*50)
    print("FINAL VALIDATION RESULTS")
    print("="*50)
    print(result)
    print("="*50)

if __name__ == "__main__":
    main()
