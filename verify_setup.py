#!/usr/bin/env python3
"""
验证项目初始化是否成功
"""

import sys
from pathlib import Path


def verify_structure():
    """验证项目结构"""
    print("验证项目结构...")
    
    required_files = [
        "llm_compression/__init__.py",
        "llm_compression/config.py",
        "llm_compression/logger.py",
        "tests/__init__.py",
        "tests/unit/__init__.py",
        "tests/property/__init__.py",
        "tests/integration/__init__.py",
        "tests/performance/__init__.py",
        "requirements.txt",
        "setup.py",
        "config.yaml",
        "README.md",
        ".gitignore",
        "pytest.ini"
    ]
    
    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)
    
    if missing:
        print(f"❌ 缺少文件: {', '.join(missing)}")
        return False
    
    print("✅ 所有必需文件存在")
    return True


def verify_imports():
    """验证模块导入"""
    print("\n验证模块导入...")
    
    try:
        from llm_compression import Config, setup_logger
        print("✅ 成功导入 Config 和 setup_logger")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    
    return True


def verify_config():
    """验证配置系统"""
    print("\n验证配置系统...")
    
    try:
        from llm_compression import Config
        
        # 测试默认配置
        config = Config()
        assert config.llm.cloud_endpoint == "http://localhost:8045"
        assert config.llm.timeout == 30.0
        assert config.compression.min_compress_length == 100
        print("✅ 默认配置正常")
        
        # 测试配置验证
        config.validate()
        print("✅ 配置验证通过")
        
        # 测试从 YAML 加载
        config_from_yaml = Config.from_yaml("config.yaml")
        config_from_yaml.validate()
        print("✅ YAML 配置加载成功")
        
    except Exception as e:
        print(f"❌ 配置系统错误: {e}")
        return False
    
    return True


def verify_logger():
    """验证日志系统"""
    print("\n验证日志系统...")
    
    try:
        from llm_compression import setup_logger
        
        logger = setup_logger("test_verify")
        logger.info("测试日志消息")
        print("✅ 日志系统正常")
        
    except Exception as e:
        print(f"❌ 日志系统错误: {e}")
        return False
    
    return True


def main():
    """主函数"""
    print("=" * 60)
    print("LLM 集成压缩系统 - 项目初始化验证")
    print("=" * 60)
    
    checks = [
        ("项目结构", verify_structure),
        ("模块导入", verify_imports),
        ("配置系统", verify_config),
        ("日志系统", verify_logger)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} 检查失败: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("验证结果汇总")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有检查通过！项目初始化成功！")
        print("=" * 60)
        print("\n下一步:")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 运行测试: pytest tests/")
        print("3. 开始开发任务 2: 实现 LLM 客户端")
        return 0
    else:
        print("⚠️  部分检查失败，请修复后重试")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
