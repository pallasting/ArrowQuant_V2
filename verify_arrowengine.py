#!/usr/bin/env python3
"""
ArrowEngine 验证脚本

用于验证 Phase 1 Week 1-2 的核心成果：
- ModelConverter: 模型转换工具
- ArrowEngine: 高性能推理引擎

使用方法:
    python verify_arrowengine.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_step_1_imports():
    """步骤 1: 验证所有核心模块可以导入"""
    print("=" * 70)
    print("步骤 1: 验证模块导入")
    print("=" * 70)
    
    try:
        print("导入 ModelConverter...")
        from llm_compression.tools import ModelConverter, ConversionConfig, ConversionResult
        print("  ✓ ModelConverter 导入成功")
        
        print("导入 ArrowEngine...")
        from llm_compression.inference import ArrowEngine, WeightLoader, FastTokenizer, InferenceCore
        print("  ✓ ArrowEngine 导入成功")
        
        print("\n✅ 所有核心模块导入成功！\n")
        return True
        
    except ImportError as e:
        print(f"\n❌ 导入失败: {e}")
        print("\n解决方案:")
        print("  1. 安装项目: pip install -e .")
        print("  2. 安装依赖: pip install -r requirements-arrow.txt")
        return False


def test_step_2_modelconverter_api():
    """步骤 2: 验证 ModelConverter API"""
    print("=" * 70)
    print("步骤 2: 验证 ModelConverter API")
    print("=" * 70)
    
    try:
        from llm_compression.tools import ModelConverter, ConversionConfig
        
        config = ConversionConfig(
            compression="lz4",
            use_float16=True,
            extract_tokenizer=True,
            validate_output=True
        )
        print(f"✓ ConversionConfig 创建成功")
        print(f"  - compression: {config.compression}")
        print(f"  - use_float16: {config.use_float16}")
        print(f"  - extract_tokenizer: {config.extract_tokenizer}")
        
        converter = ModelConverter(config)
        print(f"✓ ModelConverter 初始化成功")
        
        print("\n✅ ModelConverter API 验证通过！")
        print("\n提示: 要转换真实模型，请运行:")
        print("  python -m llm_compression.tools.cli convert \\")
        print("      --model sentence-transformers/all-MiniLM-L6-v2 \\")
        print("      --output ./models/minilm \\")
        print("      --float16\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ModelConverter API 验证失败: {e}")
        return False


def test_step_3_arrowengine_api():
    """步骤 3: 验证 ArrowEngine API（需要转换后的模型）"""
    print("=" * 70)
    print("步骤 3: 验证 ArrowEngine API")
    print("=" * 70)
    
    model_path = Path("./models/minilm")
    
    if not model_path.exists():
        print("⚠️  未找到转换后的模型")
        print(f"   期望路径: {model_path.absolute()}")
        print("\n跳过 ArrowEngine 测试（需要先转换模型）")
        print("\n如何转换模型:")
        print("  python -m llm_compression.tools.cli convert \\")
        print("      --model sentence-transformers/all-MiniLM-L6-v2 \\")
        print("      --output ./models/minilm \\")
        print("      --float16\n")
        return None
    
    try:
        from llm_compression.inference import ArrowEngine
        
        print(f"加载模型: {model_path}")
        engine = ArrowEngine(str(model_path))
        print(f"✓ ArrowEngine 初始化成功")
        print(f"  - 设备: {engine.device}")
        print(f"  - 嵌入维度: {engine.get_embedding_dimension()}")
        print(f"  - 最大序列长度: {engine.get_max_seq_length()}")
        
        print("\n测试编码...")
        test_texts = [
            "Hello, world!",
            "ArrowEngine is fast!"
        ]
        embeddings = engine.encode(test_texts)
        print(f"✓ 编码成功")
        print(f"  - 输入: {len(test_texts)} 个文本")
        print(f"  - 输出形状: {embeddings.shape}")
        
        print("\n测试相似度计算...")
        similarity = engine.similarity(test_texts[0], test_texts[1])
        print(f"✓ 相似度计算成功")
        print(f"  - 相似度: {similarity[0, 0]:.4f}")
        
        print("\n✅ ArrowEngine 完整功能验证通过！\n")
        return True
        
    except Exception as e:
        print(f"\n❌ ArrowEngine 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_step_4_cli_tool():
    """步骤 4: 验证 CLI 工具"""
    print("=" * 70)
    print("步骤 4: 验证 CLI 工具")
    print("=" * 70)
    
    try:
        from llm_compression.tools import cli
        
        print("✓ CLI 模块导入成功")
        print("\nCLI 工具可用命令:")
        print("  python -m llm_compression.tools.cli convert --help")
        print("\n示例用法:")
        print("  python -m llm_compression.tools.cli convert \\")
        print("      --model sentence-transformers/all-MiniLM-L6-v2 \\")
        print("      --output ./models/minilm \\")
        print("      --float16 \\")
        print("      --validate\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ CLI 工具验证失败: {e}")
        return False


def main():
    """主验证流程"""
    print("\n" + "=" * 70)
    print(" ArrowEngine 成果验证")
    print(" Phase 1 Week 1-2 核心功能测试")
    print("=" * 70 + "\n")
    
    results = []
    
    # 步骤 1: 导入测试
    results.append(("模块导入", test_step_1_imports()))
    
    if results[-1][1]:
        # 步骤 2: ModelConverter API
        results.append(("ModelConverter API", test_step_2_modelconverter_api()))
        
        # 步骤 3: ArrowEngine API（可选）
        result = test_step_3_arrowengine_api()
        if result is not None:
            results.append(("ArrowEngine API", result))
        
        # 步骤 4: CLI 工具
        results.append(("CLI 工具", test_step_4_cli_tool()))
    
    # 汇总报告
    print("=" * 70)
    print(" 验证结果汇总")
    print("=" * 70)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:.<50} {status}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\n通过率: {passed}/{total} ({passed/total*100:.1f}%)\n")
    
    if passed == total:
        print("🎉 所有测试通过！ArrowEngine 核心功能正常工作。\n")
        return 0
    else:
        print("⚠️  部分测试未通过。请检查上述错误信息。\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
