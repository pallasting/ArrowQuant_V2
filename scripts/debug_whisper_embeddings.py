#!/usr/bin/env python3
"""
Debug Whisper Embedding Mismatch

This script compares intermediate outputs between HuggingFace Whisper
and ArrowEngine AudioEncoder to identify where the discrepancy occurs.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_compression.logger import logger
from llm_compression.multimodal.audio_encoder import AudioEncoder, AudioConfig


def compare_tensors(name: str, hf_tensor: torch.Tensor, arrow_tensor: torch.Tensor) -> None:
    """Compare two tensors and print statistics."""
    hf_np = hf_tensor.detach().cpu().numpy().flatten()
    arrow_np = arrow_tensor.detach().cpu().numpy().flatten()
    
    # Compute statistics
    diff = hf_np - arrow_np
    abs_diff = np.abs(diff)
    
    # Cosine similarity
    hf_norm = hf_np / (np.linalg.norm(hf_np) + 1e-8)
    arrow_norm = arrow_np / (np.linalg.norm(arrow_np) + 1e-8)
    cosine_sim = np.dot(hf_norm, arrow_norm)
    
    print(f"\n{'='*70}")
    print(f"Comparing: {name}")
    print(f"{'='*70}")
    print(f"Shape: HF={hf_tensor.shape}, Arrow={arrow_tensor.shape}")
    print(f"HF   - Mean: {hf_np.mean():.6f}, Std: {hf_np.std():.6f}, Min: {hf_np.min():.6f}, Max: {hf_np.max():.6f}")
    print(f"Arrow- Mean: {arrow_np.mean():.6f}, Std: {arrow_np.std():.6f}, Min: {arrow_np.min():.6f}, Max: {arrow_np.max():.6f}")
    print(f"Diff - Mean: {diff.mean():.6f}, Std: {diff.std():.6f}, Max Abs: {abs_diff.max():.6f}")
    print(f"Cosine Similarity: {cosine_sim:.6f}")
    
    if cosine_sim < 0.9:
        print(f"[WARNING] Low similarity!")
    elif cosine_sim < 0.99:
        print(f"[CAUTION] Moderate similarity")
    else:
        print(f"[GOOD] High similarity")


def debug_whisper_conversion():
    """Debug Whisper conversion by comparing intermediate outputs."""
    
    print("\n" + "="*70)
    print("  Whisper Embedding Debug Tool")
    print("="*70)
    
    # Load models
    print("\n[1/6] Loading HuggingFace Whisper model...")
    from transformers import WhisperModel, WhisperProcessor
    
    model_name = "openai/whisper-tiny"
    hf_model = WhisperModel.from_pretrained(model_name)
    hf_processor = WhisperProcessor.from_pretrained(model_name)
    hf_model.eval()
    print(f"[OK] Loaded HuggingFace model: {model_name}")
    
    print("\n[2/6] Loading ArrowEngine AudioEncoder...")
    import json
    converted_path = Path("D:/ai-models/whisper-tiny")
    
    # Load config from metadata
    metadata_path = converted_path / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    config_dict = metadata["config"]
    config = AudioConfig(
        n_mels=config_dict["n_mels"],
        hidden_size=config_dict["hidden_size"],
        num_layers=config_dict["num_layers"],
        num_attention_heads=config_dict["num_attention_heads"],
        intermediate_size=config_dict["intermediate_size"],
        layer_norm_eps=config_dict["layer_norm_eps"],
        max_positions=config_dict["max_positions"],
    )
    
    arrow_encoder = AudioEncoder(str(converted_path), config=config)
    print(f"[OK] Loaded ArrowEngine encoder")
    print(f"   Config: hidden_size={config.hidden_size}, num_layers={config.num_layers}")
    
    # Generate test audio
    print("\n[3/6] Generating test audio...")
    sample_rate = 16000
    duration = 3.0
    num_samples = int(sample_rate * duration)
    audio = np.random.randn(num_samples).astype(np.float32)
    print(f"[OK] Generated audio: {audio.shape}, mean={audio.mean():.6f}, std={audio.std():.6f}")
    
    # Step 1: Compare mel-spectrograms
    print("\n[4/6] Comparing mel-spectrograms...")
    
    # HuggingFace mel-spectrogram
    hf_inputs = hf_processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    hf_mel = hf_inputs.input_features  # (1, 80, 3000)
    print(f"HF mel-spectrogram shape: {hf_mel.shape}")
    
    # ArrowEngine mel-spectrogram
    preprocessed = arrow_encoder.audio_processor.preprocess_batch([audio], pad_or_trim=True)
    arrow_mel = arrow_encoder.audio_processor.compute_mel_spectrogram(preprocessed[0])
    arrow_mel_tensor = torch.from_numpy(arrow_mel).unsqueeze(0)  # (1, 80, 3000)
    print(f"Arrow mel-spectrogram shape: {arrow_mel_tensor.shape}")
    
    compare_tensors("Mel-Spectrogram", hf_mel, arrow_mel_tensor)
    
    # Step 2: Compare after conv layers
    print("\n[5/6] Comparing after conv layers...")
    
    with torch.no_grad():
        # HuggingFace conv output
        hf_encoder = hf_model.encoder
        hf_conv_out = hf_encoder.conv1(hf_mel)
        hf_conv_out = torch.nn.functional.gelu(hf_conv_out)
        hf_conv_out = hf_encoder.conv2(hf_conv_out)
        hf_conv_out = torch.nn.functional.gelu(hf_conv_out)
        hf_conv_out = hf_conv_out.permute(0, 2, 1)  # (batch, time, hidden)
        print(f"HF conv output shape: {hf_conv_out.shape}")
        
        # ArrowEngine conv output
        arrow_core = arrow_encoder.core
        arrow_conv_out = arrow_core.conv1(arrow_mel_tensor)
        arrow_conv_out = torch.nn.functional.gelu(arrow_conv_out)
        arrow_conv_out = arrow_core.conv2(arrow_conv_out)
        arrow_conv_out = torch.nn.functional.gelu(arrow_conv_out)
        arrow_conv_out = arrow_conv_out.permute(0, 2, 1)  # (batch, time, hidden)
        print(f"Arrow conv output shape: {arrow_conv_out.shape}")
        
        compare_tensors("Conv Output", hf_conv_out, arrow_conv_out)
        
        # Step 3: Compare after position embedding
        print("\n[6/6] Comparing after position embedding...")
        
        # HuggingFace position embedding
        hf_pos_emb = hf_encoder.embed_positions.weight[:hf_conv_out.size(1), :]
        hf_with_pos = hf_conv_out + hf_pos_emb
        print(f"HF with position embedding shape: {hf_with_pos.shape}")
        
        # ArrowEngine position embedding
        seq_len = arrow_conv_out.size(1)
        positions = torch.arange(seq_len, device=arrow_core.device)
        arrow_pos_emb = arrow_core.position_embedding(positions)
        arrow_with_pos = arrow_conv_out + arrow_pos_emb
        print(f"Arrow with position embedding shape: {arrow_with_pos.shape}")
        
        compare_tensors("With Position Embedding", hf_with_pos, arrow_with_pos)
        
        # Step 4: Compare final embeddings
        print("\n[7/6] Comparing final embeddings...")
        
        # HuggingFace final embedding
        hf_output = hf_model.encoder(input_features=hf_mel)
        hf_final = hf_output.last_hidden_state.mean(dim=1)  # Mean pool
        print(f"HF final embedding shape: {hf_final.shape}")
        
        # ArrowEngine final embedding
        arrow_final_tensor = arrow_encoder.core(arrow_mel_tensor)
        print(f"Arrow final embedding shape: {arrow_final_tensor.shape}")
        
        compare_tensors("Final Embedding", hf_final, arrow_final_tensor)
    
    print("\n" + "="*70)
    print("  Debug Complete")
    print("="*70)


if __name__ == "__main__":
    debug_whisper_conversion()
