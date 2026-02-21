# 步骤 1: 运行 ArrowQuantizer 单元测试
pytest tests/unit/test_arrow_quantizer.py -v

# 步骤 2: 运行 GPTQ Calibrator 单元测试
pytest tests/unit/test_gptq_calibrator.py -v

# 步骤 3: 运行端到端集成测试
pytest tests/integration/test_quantization_e2e.py -v -s

# 或者运行完整的验证脚本：
python scripts/run_quantization_validation.py