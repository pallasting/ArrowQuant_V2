"""
Configuration management for EmbeddingTool

Provides utilities for loading and validating YAML configuration files
for the ArrowEngine embedding service.

Usage:
    from llm_compression.tools.config import load_config
    
    config = load_config("config/embedding_tool.yaml")
    tool = EmbeddingTool(config=config)
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

from llm_compression.tools.embedding_tool import EmbeddingConfig


class ConfigError(Exception):
    """Configuration error"""
    pass


def load_config(config_path: Optional[str] = None) -> EmbeddingConfig:
    """
    Load EmbeddingConfig from YAML file
    
    Args:
        config_path: Path to YAML configuration file.
                    If None, searches for default locations:
                    - ./config/embedding_tool.yaml
                    - ~/.llm_compression/embedding_tool.yaml
                    - /etc/llm_compression/embedding_tool.yaml
    
    Returns:
        EmbeddingConfig object
        
    Raises:
        ConfigError: If config file not found or invalid
    """
    if config_path is None:
        config_path = _find_default_config()
    
    if not os.path.exists(config_path):
        raise ConfigError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        raise ConfigError(f"Failed to parse YAML: {e}") from e
    
    return _yaml_to_config(data)


def _find_default_config() -> str:
    """Find default configuration file"""
    search_paths = [
        "./config/embedding_tool.yaml",
        os.path.expanduser("~/.llm_compression/embedding_tool.yaml"),
        "/etc/llm_compression/embedding_tool.yaml",
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return path
    
    raise ConfigError(
        f"No default config file found. Searched: {search_paths}"
    )


def _yaml_to_config(data: Dict[str, Any]) -> EmbeddingConfig:
    """Convert YAML data to EmbeddingConfig"""
    if not isinstance(data, dict):
        raise ConfigError("Invalid config: root must be a dictionary")
    
    service = data.get("service", {})
    batching = data.get("batching", {})
    cache = data.get("cache", {})
    
    return EmbeddingConfig(
        endpoint=service.get("endpoint", "http://localhost:8000"),
        timeout=float(service.get("timeout", 30.0)),
        max_retries=int(service.get("max_retries", 3)),
        batch_size=int(batching.get("batch_size", 32)),
        normalize=bool(batching.get("normalize", False)),
        enable_cache=bool(cache.get("enabled", True)),
        cache_size=int(cache.get("max_size", 1000)),
    )


def validate_config(config: EmbeddingConfig) -> None:
    """
    Validate EmbeddingConfig
    
    Args:
        config: EmbeddingConfig to validate
        
    Raises:
        ConfigError: If configuration is invalid
    """
    if config.timeout <= 0:
        raise ConfigError(f"Invalid timeout: {config.timeout} (must be > 0)")
    
    if config.max_retries < 0:
        raise ConfigError(f"Invalid max_retries: {config.max_retries} (must be >= 0)")
    
    if config.batch_size <= 0:
        raise ConfigError(f"Invalid batch_size: {config.batch_size} (must be > 0)")
    
    if config.cache_size <= 0:
        raise ConfigError(f"Invalid cache_size: {config.cache_size} (must be > 0)")
    
    if not config.endpoint:
        raise ConfigError("endpoint cannot be empty")
    
    if not config.endpoint.startswith(("http://", "https://")):
        raise ConfigError(f"Invalid endpoint URL: {config.endpoint}")


def save_config(config: EmbeddingConfig, config_path: str) -> None:
    """
    Save EmbeddingConfig to YAML file
    
    Args:
        config: EmbeddingConfig to save
        config_path: Path where to save the config file
    """
    validate_config(config)
    
    data = {
        "service": {
            "endpoint": config.endpoint,
            "timeout": config.timeout,
            "max_retries": config.max_retries,
        },
        "batching": {
            "batch_size": config.batch_size,
            "normalize": config.normalize,
        },
        "cache": {
            "enabled": config.enable_cache,
            "max_size": config.cache_size,
        },
    }
    
    os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
