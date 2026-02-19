"""
结构化 JSON 日志配置模块。

提供自定义的 JSONFormatter，将日志记录为 JSON 格式，
包含时间戳、日志级别、消息、模块、Request ID 等字段。
"""

import logging
import json
import time
from typing import Optional
from contextvars import ContextVar

# 用于存储当前请求 ID 的上下文变量
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)

class JSONFormatter(logging.Formatter):
    """自定义 JSON 日志格式化器"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # 注入 Request ID (如果存在)
        req_id = request_id_ctx.get()
        if req_id:
            log_record["request_id"] = req_id
            
        # 注入异常信息 (如果存在)
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_record, ensure_ascii=False)

def setup_logging(level: str = "INFO"):
    """配置根日志记录器"""
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除旧的处理器并添加新的 JSON 处理器
    root_logger.handlers = []
    root_logger.addHandler(handler)
    
    # 设置第三方库的日志级别，避免噪音
    logging.getLogger("uvicorn.access").handlers = []  # 禁用 uvicorn 默认访问日志
    logging.getLogger("uvicorn.access").propagate = False # 我们自己记录请求日志
    logging.getLogger("uvicorn.error").handlers = [handler]
