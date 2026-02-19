"""
FastAPI application for ArrowEngine embedding service.

Provides HTTP endpoints for:
- Text embedding generation
- Similarity computation
- Health checks
- Model information
"""

import os
import time
import uuid
import signal
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from llm_compression.inference import ArrowEngine
from llm_compression.server.models import (
    EmbedRequest,
    EmbedResponse,
    SimilarityRequest,
    SimilarityResponse,
    HealthResponse,
    InfoResponse,
)
from llm_compression.server.monitoring import (
    REQUEST_COUNT,
    INFERENCE_LATENCY,
    MODEL_LOAD_STATUS,
    REGISTRY,
)
from llm_compression.server.logging_config import setup_logging, request_id_ctx
from llm_compression.server.security import get_api_key, check_rate_limit

# 初始化结构化日志
setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

__version__ = "0.1.0"

# Global shutdown flag
_shutdown_event = asyncio.Event()
_active_requests = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for graceful startup and shutdown.
    
    Handles:
    - Model loading on startup
    - Graceful shutdown with request draining
    """
    # Startup
    logger.info("Starting ArrowEngine service...")
    
    # Register signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        _shutdown_event.set()
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    yield
    
    # Shutdown
    logger.info("Shutting down ArrowEngine service...")
    
    # Wait for active requests to complete (with timeout)
    shutdown_timeout = int(os.getenv("SHUTDOWN_TIMEOUT", "30"))
    start_time = time.time()
    
    while _active_requests > 0 and (time.time() - start_time) < shutdown_timeout:
        logger.info(f"Waiting for {_active_requests} active requests to complete...")
        await asyncio.sleep(0.5)
    
    if _active_requests > 0:
        logger.warning(f"Shutdown timeout reached with {_active_requests} active requests")
    else:
        logger.info("All requests completed, shutting down cleanly")


app = FastAPI(
    title="ArrowEngine Embedding Service",
    description="High-performance text embedding service using Arrow-optimized inference",
    version=__version__,
    lifespan=lifespan,
)

@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """
    全能中间件：处理 Request ID、Prometheus 监控、速率限制和结构化访问日志。
    """
    global _active_requests
    
    start_time = time.perf_counter()
    
    # Track active requests for graceful shutdown
    _active_requests += 1
    
    # 1. 生成并设置 Request ID
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    token = request_id_ctx.set(req_id)
    
    path = request.url.path
    method = request.method
    status = "500"
    
    try:
        # Check if shutdown is in progress
        if _shutdown_event.is_set():
            return JSONResponse(
                status_code=503,
                content={"detail": "Service is shutting down"}
            )
        
        # 2. 检查速率限制 (仅在中间件层面做简单 IP 限流)
        # 注意：429 异常会直接抛出并在下方被捕获
        await check_rate_limit(request)
        
        response = await call_next(request)
        status = str(response.status_code)
        # 将 Request ID 返回给客户端
        response.headers["X-Request-ID"] = req_id
        return response
    except Exception as e:
        # 如果是 HTTPException (如 429)，获取其状态码
        if isinstance(e, HTTPException):
            status = str(e.status_code)
        else:
            logger.error(f"Request failed: {str(e)}", exc_info=True)
        raise e
    finally:
        duration = time.perf_counter() - start_time
        
        # Decrement active request counter
        _active_requests -= 1
        
        # 3. 记录 Prometheus 指标
        REQUEST_COUNT.labels(method=method, endpoint=path, status=status).inc()
        if path in ["/embed", "/similarity"] and status == "200":
            INFERENCE_LATENCY.labels(endpoint=path).observe(duration)
            
        # 4. 记录结构化访问日志 (仅记录非健康检查接口)
        if path != "/health" and path != "/metrics":
            log_payload = {
                "method": method,
                "path": path,
                "status": status,
                "duration_ms": round(duration * 1000, 2),
                "ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown"),
            }
            logger.info("Access Log", extra=log_payload)
        
        # 清理上下文
        request_id_ctx.reset(token)

@app.get("/metrics")
async def metrics():
    """暴露 Prometheus 指标端点"""
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


_engine: Optional[ArrowEngine] = None
_model_path: Optional[str] = None


def get_engine() -> ArrowEngine:
    """Get the global ArrowEngine instance"""
    global _engine, _model_path
    
    if _engine is None:
        # Support environment variable configuration
        model_path = _model_path or os.getenv("MODEL_PATH", "./models/minilm")
        device = os.getenv("DEVICE", "cpu")
        
        try:
            _engine = ArrowEngine(model_path=model_path, device=device)
            MODEL_LOAD_STATUS.set(1)  # 标记模型已加载
        except Exception as e:
            MODEL_LOAD_STATUS.set(0)  # 标记加载失败
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model from {model_path}: {str(e)}"
            )
    
    return _engine


def init_app(model_path: str) -> None:
    """Initialize the app with a specific model path (for testing)"""
    global _model_path, _engine
    _model_path = model_path
    _engine = None


@app.post("/embed", response_model=EmbedResponse, dependencies=[Depends(get_api_key)])
async def embed(request: EmbedRequest) -> EmbedResponse:
    """
    Generate embeddings for input texts.
    
    Args:
        request: EmbedRequest with texts and optional normalize flag
        
    Returns:
        EmbedResponse with embeddings, dimension, and count
    """
    engine = get_engine()
    
    try:
        embeddings = engine.encode(request.texts)
        
        if request.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
        
        return EmbedResponse(
            embeddings=embeddings.tolist(),
            dimension=embeddings.shape[1],
            count=len(embeddings)
        )
    except Exception as e:
        # Enhanced error logging with context
        error_context = {
            "error": str(e),
            "error_type": type(e).__name__,
            "input_count": len(request.texts),
            "input_sample": request.texts[0][:50] if request.texts else None,
            "normalize": request.normalize,
            "model_state": "loaded" if engine else "not_loaded",
            "device": str(engine.device) if hasattr(engine, 'device') else None,
        }
        logger.error(
            f"Embedding generation failed: {str(e)}",
            extra=error_context,
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Embedding generation failed: {str(e)}"
        )


@app.post("/similarity", response_model=SimilarityResponse, dependencies=[Depends(get_api_key)])
async def similarity(request: SimilarityRequest) -> SimilarityResponse:
    """
    Compute similarity between text pairs.
    
    Supports two modes:
    1. Single pair: text1 and text2
    2. Multiple pairs: texts1 and texts2
    
    Args:
        request: SimilarityRequest with text pairs
        
    Returns:
        SimilarityResponse with similarity score(s)
    """
    engine = get_engine()
    
    try:
        if request.text1 is not None and request.text2 is not None:
            sim = engine.similarity(request.text1, request.text2)
            return SimilarityResponse(similarity=float(sim))
        
        elif request.texts1 is not None and request.texts2 is not None:
            emb1 = engine.encode(request.texts1)
            emb2 = engine.encode(request.texts2)
            
            emb1_norm = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
            emb2_norm = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
            
            similarities = np.sum(emb1_norm * emb2_norm, axis=1)
            
            return SimilarityResponse(similarities=similarities.tolist())
        
        else:
            raise HTTPException(
                status_code=422,
                detail="Must provide either (text1, text2) or (texts1, texts2)"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        # Enhanced error logging with context
        error_context = {
            "error": str(e),
            "error_type": type(e).__name__,
            "mode": "single" if request.text1 is not None else "batch",
            "input_count": (
                2 if request.text1 is not None 
                else len(request.texts1) if request.texts1 else 0
            ),
            "model_state": "loaded" if engine else "not_loaded",
            "device": str(engine.device) if hasattr(engine, 'device') else None,
        }
        logger.error(
            f"Similarity computation failed: {str(e)}",
            extra=error_context,
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Similarity computation failed: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with status and model loading state
    """
    try:
        engine = get_engine()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            device=str(engine.device) if hasattr(engine, 'device') else None
        )
    except Exception:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            device=None
        )


@app.get("/info", response_model=InfoResponse)
async def info() -> InfoResponse:
    """
    Get model information.
    
    Returns:
        InfoResponse with model details and server version
    """
    try:
        engine = get_engine()
        
        return InfoResponse(
            model_name=_model_path or "unknown",
            embedding_dimension=engine.get_embedding_dimension(),
            max_seq_length=engine.get_max_seq_length(),
            version=__version__,
            device=str(engine.device) if hasattr(engine, 'device') else None
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )


@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={"detail": "Not found"}
    )


@app.exception_handler(405)
async def method_not_allowed_handler(request, exc):
    """Handle 405 errors"""
    return JSONResponse(
        status_code=405,
        content={"detail": "Method not allowed"}
    )


# ============================================================================
# Server Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Support environment variable configuration
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    # Run server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level
    )
