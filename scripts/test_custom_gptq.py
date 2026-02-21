import torch
import numpy as np
from llm_compression.inference.arrow_quantizer import ArrowQuantizer, QuantizationConfig

def test_gptq_2bit():
    print("Initializing Custom 2-Bit GPTQ Quantizer...")
    # Setup the configuration
    config = QuantizationConfig(
        quant_type='int2',
        calibration_method='gptq',
        per_channel=True,
        symmetric=False
    )
    quantizer = ArrowQuantizer(config)
    
    # Create a dummy weight tensor [out_features, in_features]
    out_f, in_f = 64, 128
    weight = torch.randn(out_f, in_f).numpy().astype(np.float32)
    
    # Create dummy calibration data [samples, seq_len, hidden_dim]
    calibration_data = torch.randn(256, 16, in_f)
    print(f"Weight shape: {weight.shape}")
    print(f"Calibration data shape: {calibration_data.shape}")
    
    # Run the GPTQ algorithm
    print("Running _quantize_gptq...")
    result = quantizer._quantize_gptq(weight, [out_f, in_f], calibration_data)
    
    quantized_W = result['quantized']
    scales = result['scales']
    zero_points = result['zero_points']
    
    print("GPTQ successful!")
    print(f"Quantized shape: {quantized_W.shape}, unique values: {np.unique(quantized_W)}")
    print(f"Scales shape: {scales.shape}")
    print(f"Zero Points shape: {zero_points.shape}")
    
    # Basic check to ensure it returns -2, -1, 0, 1 for INT2
    assert quantized_W.min() >= -2 and quantized_W.max() <= 1, "Values outside INT2 range!"
    
if __name__ == "__main__":
    test_gptq_2bit()
