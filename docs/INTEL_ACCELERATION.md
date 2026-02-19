
# Intel AI Acceleration for ArrowEngine

This project supports Intel hardware acceleration through **Intel Extension for PyTorch (IPEX)** and **Advanced Matrix Extensions (AMX)**.

## How to Enable

### 1. Install Requirements
To use the AI acceleration features on Intel hardware, install the following:

```bash
# For Intel GPU (XPU) and CPU (AMX/AVX)
pip install intel-extension-for-pytorch 
```

*Note: Depending on your hardware, you may also need to install the [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html).*

### 2. Automatic Detection
The system automatically detects the best available hardware. If IPEX is installed and an Intel GPU is found, it will use the `xpu` device. If only an Intel CPU is found, it will apply `ipex.optimize()` to boost inference performance by 2-5x using AMX/AVX instructions.

## Verification
You can verify the hardware detection by running:
```bash
python diagnose_intel_hardware.py
```

## Performance Benefits
- **Intel CPU (AMX)**: Up to 3x faster inference for BERT/Transformer models.
- **Intel GPU (XPU)**: Up to 10-20x faster than CPU for large batch processing.
- **Mixed Precision**: Uses `bfloat16` or `float16` automatically if supported by the hardware.
