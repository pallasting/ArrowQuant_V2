"""
FastAPI 应用

提供健康检查和监控端点。

Requirements: 11.7
"""

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
import logging

from llm_compression.health import HealthChecker
from llm_compression.config import Config
from llm_compression.llm_client import LLMClient
from llm_compression.arrow_storage import ArrowStorage


logger = logging.getLogger(__name__)


# 创建 FastAPI 应用
app = FastAPI(
    title="LLM Compression System",
    description="Health check and monitoring API for LLM-based memory compression",
    version="1.0.0"
)


# 全局组件（在启动时初始化）
health_checker: HealthChecker = None


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化组件"""
    global health_checker
    
    try:
        # 加载配置
        config = Config.from_yaml("config.yaml")
        
        # 初始化 LLM 客户端
        llm_client = LLMClient(
            endpoint=config.llm.cloud_endpoint,
            api_key=config.llm.cloud_api_key,
            timeout=config.llm.timeout,
            max_retries=config.llm.max_retries,
            rate_limit=config.llm.rate_limit
        )
        
        # 初始化存储
        storage = ArrowStorage(
            base_path=config.storage.storage_path,
            compression_level=config.storage.compression_level
        )
        
        # 初始化健康检查器
        health_checker = HealthChecker(
            llm_client=llm_client,
            storage=storage,
            config=config
        )
        
        logger.info("FastAPI application started successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        # 创建一个基本的健康检查器
        health_checker = HealthChecker()


@app.get("/")
async def root():
    """根端点"""
    return {
        "service": "LLM Compression System",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """
    健康检查端点
    
    检查系统所有组件的健康状态：
    - LLM 客户端连接性
    - 存储可访问性
    - GPU 可用性
    - 配置有效性
    
    Returns:
        JSON 响应包含：
        - status: "healthy", "degraded", "unhealthy"
        - timestamp: 检查时间戳
        - components: 各组件详细状态
    
    Requirements: 11.7
    Property 37: 健康检查端点
    """
    if not health_checker:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": "Health checker not initialized"
            }
        )
    
    try:
        result = await health_checker.check_health()
        
        # 根据状态设置 HTTP 状态码
        status_code = {
            "healthy": 200,
            "degraded": 200,  # 降级但仍可用
            "unhealthy": 503  # 服务不可用
        }.get(result.overall_status, 503)
        
        return JSONResponse(
            status_code=status_code,
            content=result.to_dict()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": f"Health check error: {str(e)}"
            }
        )


@app.get("/health/live")
async def liveness_probe():
    """
    存活探针（Kubernetes liveness probe）
    
    简单检查应用是否运行
    """
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness_probe():
    """
    就绪探针（Kubernetes readiness probe）
    
    检查应用是否准备好接收流量
    """
    if not health_checker:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "message": "Health checker not initialized"}
        )
    
    # 快速检查关键组件
    try:
        result = await health_checker.check_health()
        
        if result.overall_status == "unhealthy":
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "message": "System unhealthy"}
            )
        
        return {"status": "ready"}
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "message": str(e)}
        )


if __name__ == "__main__":
    import uvicorn
    
    # 运行服务器
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
