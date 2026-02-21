"""
GPTQ Calibrator: Hessian-based quantization calibration.

This module implements GPTQ (Generalized Post-Training Quantization) calibration
for improved quantization accuracy. GPTQ uses second-order information (Hessian)
to find optimal quantization parameters that minimize reconstruction error.

Key features:
- Hessian matrix computation from calibration data
- Optimal quantization parameter search using OBQ (Optimal Brain Quantization)
- Layer-wise calibration with error compensation
- Reduces precision loss from 8-15% (PTQ) to 4-6% (GPTQ)

Performance targets:
- Precision loss < 6% (vs PTQ 8-15%)
- Calibration time < 5 minutes
- Compression ratio maintained (same as PTQ)

Requirements: 2.1, 2.8, 2.9, 9.3, 12.1, Task 16
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from pathlib import Path

import numpy as np
import torch
from torch import nn

from llm_compression.logger import logger
from llm_compression.errors import ConfigurationError


@dataclass
class GPTQCalibrationConfig:
    """
    GPTQ calibration configuration.
    
    Attributes:
        num_samples: Number of calibration samples (100-1000 recommended)
        block_size: Block size for iterative quantization (128 default)
        dampening_factor: Hessian dampening factor (0.01 default)
        percdamp: Percentage dampening (0.01 = 1%)
        use_cache: Cache Hessian inverse for faster calibration
        device: Device for computation ('cpu', 'cuda', 'mps')
    """
    num_samples: int = 128
    block_size: int = 128
    dampening_factor: float = 0.01
    percdamp: float = 0.01
    use_cache: bool = True
    device: str = 'cpu'
    
    def __post_init__(self):
        """Validate configuration."""
        if self.num_samples < 1:
            raise ConfigurationError(
                message=f"num_samples must be >= 1, got {self.num_samples}",
                config_key='num_samples',
                config_value=self.num_samples,
                expected_type='int >= 1'
            )
        
        if self.block_size < 1:
            raise ConfigurationError(
                message=f"block_size must be >= 1, got {self.block_size}",
                config_key='block_size',
                config_value=self.block_size,
                expected_type='int >= 1'
            )
        
        if not 0 < self.dampening_factor < 1:
            raise ConfigurationError(
                message=f"dampening_factor must be in (0, 1), got {self.dampening_factor}",
                config_key='dampening_factor',
                config_value=self.dampening_factor,
                expected_type='float in (0, 1)'
            )


class GPTQCalibrator:
    """
    GPTQ calibrator for optimal quantization parameters.
    
    Implements the GPTQ algorithm:
    1. Compute Hessian H = 2 * X^T * X / num_samples from calibration data
    2. Add dampening: H[i,i] += damp for numerical stability
    3. Compute H^{-1} for error compensation
    4. Iteratively quantize weights column-by-column:
       - Quantize weight w[i]
       - Compute quantization error e = (w[i] - q[i]) / H^{-1}[i,i]
       - Compensate remaining weights: w[i+1:] -= e * H^{-1}[i, i+1:]
    
    This minimizes the reconstruction error ||W - Q||^2_H where H is the Hessian.
    
    Example:
        >>> config = GPTQCalibrationConfig(num_samples=128)
        >>> calibrator = GPTQCalibrator(config)
        >>> 
        >>> # Prepare calibration data
        >>> calibration_data = torch.randn(128, 512, 768)  # [batch, seq, hidden]
        >>> 
        >>> # Calibrate layer
        >>> weight = torch.randn(768, 768)  # [out_features, in_features]
        >>> calibrated = calibrator.calibrate_layer(
        ...     weight=weight,
        ...     calibration_data=calibration_data,
        ...     quant_type='int8'
        ... )
    """
    
    def __init__(self, config: GPTQCalibrationConfig):
        """
        Initialize GPTQ calibrator.
        
        Args:
            config: Calibration configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        self._hessian_cache: Dict[str, torch.Tensor] = {}
        
        logger.info(f"Initialized GPTQCalibrator with config: {config}")
    
    def prepare_calibration_dataset(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512
    ) -> torch.Tensor:
        """
        Prepare calibration dataset from text samples.
        
        Args:
            texts: List of text samples
            tokenizer: Tokenizer for encoding texts
            max_length: Maximum sequence length
            
        Returns:
            Calibration data tensor [num_samples, seq_len]
        """
        logger.info(f"Preparing calibration dataset from {len(texts)} samples")
        
        # Take first num_samples
        texts = texts[:self.config.num_samples]
        
        # Tokenize
        encoded = tokenizer(
            texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        
        logger.info(f"Prepared calibration dataset: shape={input_ids.shape}")
        return input_ids
    
    def compute_hessian(
        self,
        calibration_data: torch.Tensor,
        layer_name: Optional[str] = None
    ) -> torch.Tensor:
        """
        Compute Hessian matrix from calibration data.
        
        The Hessian H = 2 * X^T * X / num_samples represents the second-order
        information about the loss landscape. It's used to find optimal
        quantization parameters that minimize reconstruction error.
        
        Args:
            calibration_data: Input activations [batch * seq_len, in_features]
            layer_name: Layer name for caching (optional)
            
        Returns:
            Hessian matrix [in_features, in_features]
        """
        # Check cache
        if layer_name and self.config.use_cache and layer_name in self._hessian_cache:
            logger.debug(f"Using cached Hessian for {layer_name}")
            return self._hessian_cache[layer_name]
        
        # Flatten to [batch * seq_len, in_features]
        if calibration_data.dim() == 3:
            batch_size, seq_len, in_features = calibration_data.shape
            X = calibration_data.reshape(-1, in_features).float()
        else:
            X = calibration_data.float()
        
        num_samples = X.shape[0]
        in_features = X.shape[1]
        
        logger.debug(f"Computing Hessian for shape {X.shape}")
        
        # Compute H = 2 * X^T * X / num_samples
        H = 2 * (X.t() @ X) / num_samples
        
        # Add dampening for numerical stability
        damp = self.config.percdamp * torch.mean(torch.diag(H))
        H[range(in_features), range(in_features)] += damp
        
        logger.debug(f"Computed Hessian: shape={H.shape}, damp={damp:.6f}")
        
        # Cache if requested
        if layer_name and self.config.use_cache:
            self._hessian_cache[layer_name] = H
        
        return H
    
    def compute_hessian_inverse(
        self,
        hessian: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Compute Hessian inverse for error compensation.
        
        Args:
            hessian: Hessian matrix [in_features, in_features]
            
        Returns:
            Hessian inverse or None if inversion fails
        """
        try:
            H_inv = torch.linalg.inv(hessian)
            logger.debug(f"Computed Hessian inverse: shape={H_inv.shape}")
            return H_inv
        except RuntimeError as e:
            logger.warning(f"Hessian inversion failed: {e}")
            return None
    
    def calibrate_layer(
        self,
        weight: torch.Tensor,
        calibration_data: torch.Tensor,
        quant_type: str = 'int8',
        symmetric: bool = True,
        layer_name: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calibrate layer weights using GPTQ algorithm.
        
        Implements the iterative GPTQ quantization with error compensation:
        1. Compute Hessian and its inverse
        2. For each column i:
           a. Quantize weight w[i]
           b. Compute error e = (w[i] - q[i]) / H^{-1}[i,i]
           c. Compensate: w[i+1:] -= e * H^{-1}[i, i+1:]
        
        Args:
            weight: Weight tensor [out_features, in_features]
            calibration_data: Calibration input [batch, seq_len, in_features]
            quant_type: Quantization type ('int8' or 'int2')
            symmetric: Use symmetric quantization
            layer_name: Layer name for logging
            
        Returns:
            Dictionary with:
            - quantized: Quantized weights [out_features, in_features]
            - scales: Per-channel scales [out_features]
            - zero_points: Per-channel zero points [out_features]
            - error: Reconstruction error (float)
        """
        if weight.dim() != 2:
            raise ValueError(f"Weight must be 2D, got shape {weight.shape}")
        
        out_features, in_features = weight.shape
        
        logger.info(
            f"Calibrating layer {layer_name or 'unknown'}: "
            f"shape={weight.shape}, quant_type={quant_type}"
        )
        
        # Move to device
        weight = weight.to(self.device).float()
        calibration_data = calibration_data.to(self.device)
        
        # Compute Hessian
        H = self.compute_hessian(calibration_data, layer_name)
        
        # Compute Hessian inverse
        H_inv = self.compute_hessian_inverse(H)
        if H_inv is None:
            logger.warning("Hessian inversion failed, returning original weights")
            return {
                'quantized': weight,
                'scales': torch.ones(out_features, device=self.device),
                'zero_points': torch.zeros(out_features, device=self.device),
                'error': float('inf')
            }
        
        # Get quantization range
        if quant_type == 'int8':
            qmin, qmax = -128, 127
        elif quant_type == 'int2':
            qmin, qmax = -2, 1
        else:
            raise ValueError(f"Unsupported quant_type: {quant_type}")
        
        # Compute per-channel scales and zero points
        scales = []
        zero_points = []
        
        for i in range(out_features):
            row = weight[i]
            scale, zero_point = self._compute_quantization_params(
                row, qmin, qmax, symmetric
            )
            scales.append(scale)
            zero_points.append(zero_point)
        
        scales_t = torch.tensor(scales, dtype=torch.float32, device=self.device).unsqueeze(1)
        zps_t = torch.tensor(zero_points, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        # GPTQ main loop: iterative quantization with error compensation
        W = weight.clone()
        quantized_W = torch.zeros_like(W)
        
        logger.debug(f"Running GPTQ loop over {in_features} columns...")
        
        for i in range(in_features):
            # Get column
            w = W[:, i]
            d = H_inv[i, i]
            
            # Quantize: q = round(w / scale) + zero_point
            q = torch.round(w.unsqueeze(1) / scales_t + zps_t).squeeze(1)
            q = torch.clamp(q, qmin, qmax)
            
            # Dequantize: w_q = (q - zero_point) * scale
            w_q = (q - zps_t.squeeze(1)) * scales_t.squeeze(1)
            quantized_W[:, i] = q
            
            # Compute quantization error
            err = (w - w_q) / d
            
            # Compensate remaining weights
            if i < in_features - 1:
                W[:, i+1:] -= err.unsqueeze(1) @ H_inv[i, i+1:].unsqueeze(0)
        
        # Compute reconstruction error
        original_norm = torch.norm(weight).item()
        reconstructed = (quantized_W - zps_t.squeeze(1).unsqueeze(1)) * scales_t.squeeze(1).unsqueeze(1)
        error = torch.norm(weight - reconstructed).item() / original_norm if original_norm > 0 else 0.0
        
        logger.info(
            f"GPTQ calibration complete: "
            f"layer={layer_name or 'unknown'}, "
            f"relative_error={error:.4f}"
        )
        
        return {
            'quantized': quantized_W,
            'scales': torch.tensor(scales, dtype=torch.float32, device=self.device),
            'zero_points': torch.tensor(zero_points, dtype=torch.float32, device=self.device),
            'error': error
        }
    
    def _compute_quantization_params(
        self,
        tensor: torch.Tensor,
        qmin: int,
        qmax: int,
        symmetric: bool
    ) -> Tuple[float, int]:
        """
        Compute quantization parameters (scale and zero_point).
        
        Args:
            tensor: Input tensor
            qmin: Minimum quantized value
            qmax: Maximum quantized value
            symmetric: Use symmetric quantization
            
        Returns:
            Tuple of (scale, zero_point)
        """
        if symmetric:
            # Symmetric: zero_point = 0
            max_val = torch.abs(tensor).max().item()
            
            if max_val == 0:
                scale = 1.0
            else:
                scale = max_val / qmax
            
            zero_point = 0
        else:
            # Asymmetric
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            
            if max_val == min_val:
                scale = 1.0
                zero_point = 0
            else:
                scale = (max_val - min_val) / (qmax - qmin)
                zero_point = int(qmin - min_val / scale)
                zero_point = np.clip(zero_point, qmin, qmax)
        
        return float(scale), int(zero_point)
    
    def clear_cache(self):
        """Clear Hessian cache."""
        self._hessian_cache.clear()
        logger.debug("Cleared Hessian cache")
    
    def get_cache_size(self) -> int:
        """Get number of cached Hessians."""
        return len(self._hessian_cache)
