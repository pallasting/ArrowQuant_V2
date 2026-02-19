# Model Conversion Tools Consolidation - Execution Plan

## Overview

This document outlines the detailed execution plan for consolidating all model conversion tools into a unified ModelConverter system.

**Goal**: Create a single, well-tested, production-ready model converter that supports BERT, CLIP, and Whisper models.

**Timeline**: 2-3 weeks  
**Priority**: High  
**Status**: Planning Complete, Ready for Implementation

---

## Phase 1: Foundation (Days 1-5)

### Task 1.1: Extend ModelConverter for CLIP Support

**Objective**: Add CLIP conversion capability to ModelConverter

**Implementation**:

```python
# In llm_compression/tools/model_converter.py

def _convert_clip(
    self,
    model_name: str,
    output_dir: Path
) -> ConversionResult:
    """
    Convert CLIP model to Arrow/Parquet format.
    
    Args:
        model_name: HuggingFace CLIP model name
        output_dir: Output directory
        
    Returns:
        ConversionResult with conversion details
    """
    from transformers import CLIPModel, CLIPProcessor
    
    # Load model
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    config = model.config
    
    # Extract vision encoder weights only
    weights = self._extract_clip_weights(model)
    
    # Map keys if needed
    weights = self._map_clip_keys(weights)
    
    # Continue with standard conversion pipeline
    # (optimize, convert to arrow, validate, metadata)
    ...
```

**Files to Modify**:
- `llm_compression/tools/model_converter.py`

**New Methods**:
- `_convert_clip()`
- `_extract_clip_weights()`
- `_map_clip_keys()`
- `_generate_clip_metadata()`

**Tests to Add**:
- `tests/unit/tools/test_clip_conversion.py`

**Acceptance Criteria**:
- ✅ CLIP models convert successfully
- ✅ Output format matches existing schema
- ✅ Validation passes
- ✅ Metadata includes CLIP-specific config
- ✅ Test coverage >85%

---

### Task 1.2: Extend ModelConverter for Whisper Support

**Objective**: Add Whisper conversion capability to ModelConverter

**Implementation**:

```python
def _convert_whisper(
    self,
    model_name: str,
    output_dir: Path
) -> ConversionResult:
    """
    Convert Whisper model to Arrow/Parquet format.
    
    Args:
        model_name: HuggingFace Whisper model name
        output_dir: Output directory
        
    Returns:
        ConversionResult with conversion details
    """
    from transformers import WhisperModel, WhisperProcessor
    
    # Load model
    model = WhisperModel.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name)
    config = model.config
    
    # Extract encoder weights only (not decoder)
    weights = self._extract_whisper_weights(model)
    
    # Map keys (embed_positions → position_embedding)
    weights = self._map_whisper_keys(weights)
    
    # Continue with standard conversion pipeline
    ...
```

**Files to Modify**:
- `llm_compression/tools/model_converter.py`

**New Methods**:
- `_convert_whisper()`
- `_extract_whisper_weights()`
- `_map_whisper_keys()`
- `_generate_whisper_metadata()`

**Tests to Add**:
- `tests/unit/tools/test_whisper_conversion.py`

**Acceptance Criteria**:
- ✅ Whisper models convert successfully
- ✅ Only encoder weights extracted
- ✅ Key mapping works correctly
- ✅ Validation passes
- ✅ Test coverage >85%

---

### Task 1.3: Add Model Type Auto-Detection

**Objective**: Automatically detect model type from model name or config

**Implementation**:

```python
def _detect_model_type(self, model_name: str) -> str:
    """
    Auto-detect model type from model name or config.
    
    Args:
        model_name: Model name or path
        
    Returns:
        Model type: "bert", "clip", "whisper", or "unknown"
    """
    # Check model name patterns
    if "clip" in model_name.lower():
        return "clip"
    elif "whisper" in model_name.lower():
        return "whisper"
    elif "bert" in model_name.lower() or "sentence-transformers" in model_name.lower():
        return "bert"
    
    # Try loading config
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        
        if hasattr(config, 'model_type'):
            if config.model_type == "clip":
                return "clip"
            elif config.model_type == "whisper":
                return "whisper"
            elif config.model_type in ["bert", "roberta", "distilbert"]:
                return "bert"
    except:
        pass
    
    return "unknown"
```

**Files to Modify**:
- `llm_compression/tools/model_converter.py`

**Tests to Add**:
- Test auto-detection for various model names
- Test fallback to config inspection
- Test unknown model handling

**Acceptance Criteria**:
- ✅ Correctly detects BERT models
- ✅ Correctly detects CLIP models
- ✅ Correctly detects Whisper models
- ✅ Handles unknown models gracefully

---

### Task 1.4: Update Main convert() Method

**Objective**: Route to appropriate converter based on model type

**Implementation**:

```python
def convert(
    self,
    model_name_or_path: str,
    output_dir: str,
    model_type: str = "auto"
) -> ConversionResult:
    """
    Convert a model to Arrow/Parquet format.
    
    Args:
        model_name_or_path: HuggingFace model name or local path
        output_dir: Directory to save converted model
        model_type: Type of model ("auto", "bert", "clip", "whisper")
        
    Returns:
        ConversionResult: Detailed conversion result
    """
    # Auto-detect if needed
    if model_type == "auto":
        model_type = self._detect_model_type(model_name_or_path)
        logger.info(f"Auto-detected model type: {model_type}")
    
    # Route to appropriate converter
    if model_type == "clip":
        return self._convert_clip(model_name_or_path, Path(output_dir))
    elif model_type == "whisper":
        return self._convert_whisper(model_name_or_path, Path(output_dir))
    elif model_type in ["bert", "sentence-transformers", "transformers"]:
        return self._convert_bert(model_name_or_path, output_dir)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
```

**Files to Modify**:
- `llm_compression/tools/model_converter.py`

**Acceptance Criteria**:
- ✅ Backward compatible with existing BERT conversions
- ✅ Supports explicit model_type specification
- ✅ Supports auto-detection
- ✅ Clear error messages for unsupported types

---

### Task 1.5: Standardize on Zstandard Compression

**Objective**: Use Zstandard compression for all models

**Implementation**:

```python
@dataclass
class ConversionConfig:
    """Configuration for model conversion process."""
    
    compression: str = "zstd"  # Changed from "lz4"
    compression_level: int = 3  # Zstd compression level (1-22)
    use_float16: bool = True
    extract_tokenizer: bool = True
    validate_output: bool = True
```

**Files to Modify**:
- `llm_compression/tools/model_converter.py`

**Migration**:
- Update default compression in ConversionConfig
- Update _convert_to_arrow() to use zstd
- Add compression_level parameter

**Acceptance Criteria**:
- ✅ All conversions use Zstandard by default
- ✅ Compression level configurable
- ✅ Backward compatible (can still read LZ4 files)
- ✅ Better compression ratios achieved

---

## Phase 2: Integration (Days 6-10)

### Task 2.1: Create Unified CLI Script

**Objective**: Single command-line interface for all conversions

**File**: `scripts/convert_model.py`

**Implementation**:

```python
#!/usr/bin/env python3
"""
Unified Model Converter - Convert any supported model to Arrow/Parquet format

Supports:
- BERT and sentence-transformers models
- CLIP vision models
- Whisper audio models

Usage:
    python scripts/convert_model.py --model <name> --output <dir> [--type auto]

Examples:
    # Auto-detect model type
    python scripts/convert_model.py \\
        --model openai/clip-vit-base-patch32 \\
        --output models/clip-vit-b32
    
    # Explicit model type
    python scripts/convert_model.py \\
        --model openai/whisper-base \\
        --output models/whisper-base \\
        --type whisper
"""

import argparse
from pathlib import Path

from llm_compression.tools import ModelConverter, ConversionConfig
from llm_compression.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert models to Arrow/Parquet format"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or local path"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for converted model"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="auto",
        choices=["auto", "bert", "clip", "whisper"],
        help="Model type (default: auto-detect)"
    )
    parser.add_argument(
        "--float16",
        action="store_true",
        default=True,
        help="Convert to float16 (default: True)"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation"
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="zstd",
        choices=["zstd", "lz4", "snappy"],
        help="Compression algorithm (default: zstd)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create config
    config = ConversionConfig(
        compression=args.compression,
        use_float16=args.float16,
        validate_output=not args.no_validate
    )
    
    # Create converter
    converter = ModelConverter(config)
    
    # Convert
    logger.info(f"Converting {args.model} to {args.output}")
    result = converter.convert(
        model_name_or_path=args.model,
        output_dir=args.output,
        model_type=args.type
    )
    
    # Report results
    if result.success:
        logger.info("=" * 60)
        logger.info("Conversion Summary:")
        logger.info(f"  Model: {result.model_name}")
        logger.info(f"  Output: {result.output_dir}")
        logger.info(f"  Parameters: {result.total_parameters:,}")
        logger.info(f"  File size: {result.file_size_mb:.2f} MB")
        logger.info(f"  Compression: {result.compression_ratio:.2f}x")
        logger.info(f"  Time: {result.conversion_time_sec:.2f} seconds")
        logger.info(f"  Validation: {'PASSED' if result.validation_passed else 'SKIPPED'}")
        logger.info("=" * 60)
        logger.info("SUCCESS: Model converted successfully")
        return 0
    else:
        logger.error(f"Conversion failed: {result.error_message}")
        return 1


if __name__ == "__main__":
    exit(main())
```

**Acceptance Criteria**:
- ✅ Single entry point for all conversions
- ✅ Auto-detection works
- ✅ Explicit type specification works
- ✅ Clear help messages
- ✅ Comprehensive error handling

---

### Task 2.2: Update Documentation

**Files to Update**:
1. `docs/QUICKSTART_MULTIMODAL.md`
2. `docs/API_REFERENCE_COMPLETE.md`
3. `README.md`
4. `AGENTS.md`

**Changes**:
- Add unified converter usage examples
- Update model conversion sections
- Add auto-detection examples
- Document all supported model types

**Example Update**:

```markdown
## Model Conversion

Convert any supported model using the unified converter:

```bash
# Auto-detect model type
python scripts/convert_model.py \\
    --model openai/clip-vit-base-patch32 \\
    --output D:/ai-models/clip-vit-b32

# Or use specific scripts (backward compatible)
python scripts/convert_clip_to_parquet.py \\
    --model openai/clip-vit-base-patch32 \\
    --output D:/ai-models/clip-vit-b32
```

**Acceptance Criteria**:
- ✅ All documentation updated
- ✅ Examples tested and working
- ✅ Backward compatibility noted
- ✅ Clear migration guide

---

### Task 2.3: Deprecate Standalone Scripts

**Objective**: Mark standalone scripts as deprecated but keep for backward compatibility

**Files to Update**:
- `scripts/convert_clip_to_parquet.py`
- `scripts/convert_whisper_to_parquet.py`

**Changes**:

```python
#!/usr/bin/env python3
"""
CLIP Model Converter - DEPRECATED

⚠️ DEPRECATION NOTICE:
This script is deprecated. Please use the unified converter instead:

    python scripts/convert_model.py --model <name> --output <dir>

This script will be removed in version 2.0.0.
For now, it remains for backward compatibility.
"""

import warnings

warnings.warn(
    "convert_clip_to_parquet.py is deprecated. "
    "Use scripts/convert_model.py instead.",
    DeprecationWarning,
    stacklevel=2
)

# Rest of the script remains unchanged for backward compatibility
...
```

**Acceptance Criteria**:
- ✅ Deprecation warnings added
- ✅ Scripts still functional
- ✅ Clear migration path provided
- ✅ Removal timeline communicated

---

### Task 2.4: Remove Legacy Code

**Objective**: Delete deprecated convert_clip.py

**Files to Delete**:
- `llm_compression/tools/convert_clip.py`

**Verification**:
- Check for any imports or references
- Update __init__.py if needed
- Remove from documentation

**Acceptance Criteria**:
- ✅ File deleted
- ✅ No broken imports
- ✅ Tests still pass
- ✅ Documentation updated

---

## Phase 3: Enhancement (Days 11-15)

### Task 3.1: Add Progress Bars

**Objective**: Show conversion progress for better UX

**Implementation**:

```python
from tqdm import tqdm

def _convert_to_arrow(self, weights, output_dir, model_name):
    """Convert weights with progress bar."""
    logger.info("Converting weights to Parquet format...")
    
    # Prepare data with progress bar
    layer_names = []
    shapes = []
    dtypes = []
    data_blobs = []
    num_params_list = []
    
    with tqdm(total=len(weights), desc="Processing layers") as pbar:
        for layer_name, tensor in weights.items():
            # Process layer
            layer_names.append(layer_name)
            shapes.append(list(tensor.shape))
            dtypes.append(str(tensor.dtype))
            data_blobs.append(tensor.numpy().tobytes())
            num_params_list.append(tensor.numel())
            pbar.update(1)
    
    # Create and write table
    ...
```

**Acceptance Criteria**:
- ✅ Progress bars for long operations
- ✅ ETA displayed
- ✅ Can be disabled for scripting

---

### Task 3.2: Add Comprehensive Tests

**Test Files**:
- `tests/unit/tools/test_clip_conversion.py`
- `tests/unit/tools/test_whisper_conversion.py`
- `tests/integration/test_unified_converter.py`

**Test Coverage**:
- CLIP conversion end-to-end
- Whisper conversion end-to-end
- Auto-detection logic
- Error handling
- Validation logic
- Metadata generation

**Target**: >90% coverage for model_converter.py

---

### Task 3.3: Performance Benchmarking

**Objective**: Ensure no performance regression

**Benchmarks**:
1. Conversion time (before vs after)
2. Compression ratio (LZ4 vs Zstd)
3. Memory usage
4. File size

**Acceptance Criteria**:
- ✅ Conversion time ≤ baseline
- ✅ Compression ratio ≥ baseline
- ✅ Memory usage ≤ baseline

---

## Testing Strategy

### Unit Tests

```python
# tests/unit/tools/test_clip_conversion.py

def test_clip_conversion():
    """Test CLIP model conversion."""
    converter = ModelConverter()
    result = converter.convert(
        model_name_or_path="openai/clip-vit-base-patch32",
        output_dir="test_output/clip",
        model_type="clip"
    )
    
    assert result.success
    assert result.parquet_path.exists()
    assert result.metadata_path.exists()
    assert result.validation_passed
    assert result.total_parameters > 0
```

### Integration Tests

```python
# tests/integration/test_unified_converter.py

def test_auto_detection():
    """Test auto-detection of model types."""
    converter = ModelConverter()
    
    # Test CLIP detection
    result = converter.convert(
        model_name_or_path="openai/clip-vit-base-patch32",
        output_dir="test_output/clip_auto",
        model_type="auto"
    )
    assert result.success
    
    # Test Whisper detection
    result = converter.convert(
        model_name_or_path="openai/whisper-base",
        output_dir="test_output/whisper_auto",
        model_type="auto"
    )
    assert result.success
```

---

## Success Metrics

### Quantitative

- ✅ Code reduction: 1170 → 650 lines (45% reduction)
- ✅ Test coverage: 28% → 90%+
- ✅ Conversion time: ≤ baseline
- ✅ Compression ratio: ≥ baseline
- ✅ All existing conversions still work

### Qualitative

- ✅ Single source of truth
- ✅ Consistent behavior across model types
- ✅ Better user experience
- ✅ Easier maintenance
- ✅ Clear documentation

---

## Risk Mitigation

### Risk 1: Breaking Changes

**Mitigation**:
- Keep standalone scripts for 6 months
- Add deprecation warnings
- Provide clear migration guide
- Extensive testing

### Risk 2: Performance Regression

**Mitigation**:
- Benchmark before/after
- Profile critical paths
- Optimize if needed

### Risk 3: Bugs in New Code

**Mitigation**:
- Comprehensive test suite
- Validation against existing conversions
- Gradual rollout

---

## Timeline

| Week | Tasks | Deliverables |
|------|-------|--------------|
| Week 1 | Phase 1 (Tasks 1.1-1.5) | Extended ModelConverter |
| Week 2 | Phase 2 (Tasks 2.1-2.4) | Unified CLI, Updated docs |
| Week 3 | Phase 3 (Tasks 3.1-3.3) | Enhancements, Tests |

---

## Next Steps

1. ✅ Complete audit (DONE)
2. ⏳ Review and approve plan
3. ⏳ Begin Phase 1 implementation
4. ⏳ Continuous testing and validation
5. ⏳ Documentation updates
6. ⏳ Final review and deployment

---

## Conclusion

This consolidation plan provides a clear path to unifying all model conversion tools while maintaining backward compatibility and improving code quality. The phased approach minimizes risk and ensures thorough testing at each stage.

**Recommendation**: Proceed with implementation starting with Phase 1.

