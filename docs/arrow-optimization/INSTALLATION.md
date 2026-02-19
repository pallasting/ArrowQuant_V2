# Arrow Optimization Setup - Installation Verification

## Installation Status

This document tracks the installation status of Arrow optimization dependencies.

## Dependencies Status

### Core Dependencies (from requirements.txt)
- ✅ pyarrow>=14.0.0 (Already installed)
- ✅ torch>=2.0.0 (Already installed)
- ✅ numpy>=1.24.0 (Already installed)
- ✅ pandas>=2.0.0 (Already installed)

### Arrow Optimization Dependencies (from requirements-arrow.txt)
To install, run:
```bash
pip install -r requirements-arrow.txt
```

Dependencies to be installed:
- tokenizers>=0.15.0 (Rust tokenizer - 10-20x faster)
- fastapi>=0.104.0 (API framework)
- uvicorn[standard]>=0.24.0 (ASGI server)
- pydantic>=2.0.0 (Data validation)
- prometheus-client>=0.19.0 (Monitoring)
- python-json-logger>=2.0.0 (Structured logging)
- pytest-benchmark>=4.0.0 (Performance testing)
- locust>=2.15.0 (Load testing)
- httpx>=0.25.0 (Async HTTP client)

## Installation Instructions

### Option 1: Automatic Setup (Recommended)
```bash
bash scripts/arrow/setup_dev_env.sh
```

### Option 2: Manual Installation
```bash
# Install Arrow optimization dependencies
pip install -r requirements-arrow.txt

# Verify installation
python -c "import tokenizers; print(f'✅ Tokenizers {tokenizers.__version__}')"
python -c "import fastapi; print(f'✅ FastAPI {fastapi.__version__}')"
python -c "import uvicorn; print(f'✅ Uvicorn {uvicorn.__version__}')"
```

### Option 3: Virtual Environment (Isolated)
```bash
# Create virtual environment
python -m venv venv-arrow

# Activate
source venv-arrow/bin/activate  # Linux/Mac
# or
venv-arrow\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-arrow.txt
```

## Verification

Run verification script:
```bash
python scripts/arrow/verify_installation.py
```

Expected output:
```
✅ Python 3.10.x
✅ PyArrow 14.x.x
✅ Tokenizers 0.15.x
✅ FastAPI 0.104.x
✅ Uvicorn 0.24.x
```

## Next Steps

After installation:
1. Convert a model: `python -m llm_compression.tools.convert_model <model-name>`
2. Run tests: `pytest tests/unit/`
3. Start development: See docs/arrow-optimization/TASKS.md

## Troubleshooting

### Issue: "tokenizers not found"
```bash
pip install tokenizers>=0.15.0
```

### Issue: "fastapi not found"
```bash
pip install fastapi>=0.104.0 uvicorn[standard]>=0.24.0
```

### Issue: Permission denied
```bash
pip install --user -r requirements-arrow.txt
```

---

**Status**: Dependencies ready for installation
**Last Updated**: 2026-02-17
