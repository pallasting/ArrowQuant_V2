"""
日志模块单元测试
"""

import logging
import tempfile
from pathlib import Path

from llm_compression.logger import setup_logger


class TestLogger:
    """日志系统测试"""
    
    def test_setup_logger_default(self):
        """测试默认日志设置"""
        logger = setup_logger(name="test_default")
        
        assert logger.name == "test_default"
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0
    
    def test_setup_logger_with_level(self):
        """测试指定日志级别"""
        logger = setup_logger(name="test_level", level="DEBUG")
        
        assert logger.level == logging.DEBUG
    
    def test_setup_logger_with_file(self):
        """测试文件日志"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = "test.log"
            logger = setup_logger(
                name="test_file",
                level="INFO",
                log_file=log_file,
                log_dir=temp_dir
            )
            
            # 写入日志
            logger.info("Test message")
            
            # 验证文件存在
            log_path = Path(temp_dir) / log_file
            assert log_path.exists()
            
            # 验证内容
            content = log_path.read_text()
            assert "Test message" in content
    
    def test_logger_no_duplicate_handlers(self):
        """测试不会重复添加处理器"""
        logger1 = setup_logger(name="test_dup")
        handler_count1 = len(logger1.handlers)
        
        logger2 = setup_logger(name="test_dup")
        handler_count2 = len(logger2.handlers)
        
        # 应该是同一个 logger，处理器数量不变
        assert handler_count1 == handler_count2
        assert logger1 is logger2
