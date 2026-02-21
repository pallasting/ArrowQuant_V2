#!/usr/bin/env python3
"""
Generate Calibration Data for GPTQ Quantization

This script runs inference on a HuggingFace model using a set of calibration texts,
hooks into the Linear layers, and saves their input activations to a `.pt` file.
This data is required for the Hessian-based ArrowQuant GPTQ implementation.

Usage:
    python scripts/generate_calibration_data.py \
        --model sentence-transformers/all-MiniLM-L6-v2 \
        --output calibration_data_minilm.pt \
        --samples 128
"""

import argparse
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

def create_parser():
    parser = argparse.ArgumentParser(description="Generate layer-wise calibration data for GPTQ")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name or path")
    parser.add_argument("--output", type=str, required=True, help="Output path for the .pt file")
    parser.add_argument("--samples", type=int, default=128, help="Number of calibration sentences to use")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length for tokenizer")
    return parser

def get_calibration_texts(num_samples: int):
    """Generate or load calibration texts. Currently uses wikipedia-like sentences."""
    # A tiny built-in corpus. For exact production use, load from a real HF dataset like 'wikitext'.
    base_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the technology industry.",
        "Quantization reduces the memory footprint of large language models.",
        "New breakthroughs in deep learning happen every day.",
        "Diffusion models are becoming the unified architecture for multimodal generation.",
        "Tensors flow through the computation graph during the forward pass.",
        "Self-attention mechanisms allow Transformers to contextualize sequences.",
        "Hessian matrices encode the second-order derivatives of the loss function.",
        "Zero-copy memory access via Apache Arrow significantly speeds up model loading.",
        "Optimal Brain Damage and Optimal Brain Surgeon paved the way for modern pruning."
    ]
    
    # Repeat and slice to get the requested number of samples
    texts = (base_texts * ((num_samples // len(base_texts)) + 1))[:num_samples]
    return texts

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    print(f"Loading model and tokenizer for: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.eval()
    
    print(f"Preparing {args.samples} calibration samples...")
    texts = get_calibration_texts(args.samples)
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
    
    # Dictionary to store the captured inputs
    calibration_data = {}
    hooks = []
    
    # Define the hook function
    def make_hook(name):
        def hook(module, input, output):
            # input is a tuple of (activation, )
            activation = input[0].detach()
            
            # If we process in batches, we would concatenate.
            # Here we just assume everything fits in one batch for simplicity,
            # or we overwrite. We should append if doing mini-batches.
            # For this script we run the entire batch at once.
            if name not in calibration_data:
                calibration_data[name] = activation
            else:
                calibration_data[name] = torch.cat([calibration_data[name], activation], dim=0)
        return hook
    
    print("Registering hooks on Linear layers...")
    hooked_layers = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # arrow_quantizer uses base name (without .weight) to match linear layer
            hooks.append(module.register_forward_hook(make_hook(name)))
            hooked_layers += 1
            
    print(f"Hooked {hooked_layers} layers. Running forward pass...")
    with torch.no_grad():
        # In a real scenario with thousands of samples, process in batches.
        # But for 128 samples of length 128, a single forward pass on CPU is fine.
        model(**inputs)
        
    print(f"Captured calibration data for {len(calibration_data)} layers.")
    
    # Cleanup hooks
    for h in hooks:
        h.remove()
        
    # Save the data
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving calibration data to {output_path}...")
    torch.save(calibration_data, output_path)
    
    print("Done! 🎉")

if __name__ == "__main__":
    main()
