"""
Prometheus metrics for monitoring.
"""
from typing import Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import time

from llm_compression.logger import logger


@dataclass
class MetricsCollector:
    """Collect and expose Prometheus metrics."""
    
    # Request metrics
    request_count: int = 0
    request_duration_sum: float = 0.0
    request_errors: int = 0
    
    # Quantization metrics
    quantization_count: int = 0
    quantization_duration_sum: float = 0.0
    quantization_errors: int = 0
    
    # Compression metrics
    total_bytes_compressed: int = 0
    total_bytes_original: int = 0
    
    # System metrics
    start_time: float = field(default_factory=time.time)
    
    def record_request(self, duration: float, error: bool = False) -> None:
        """Record API request metrics."""
        self.request_count += 1
        self.request_duration_sum += duration
        if error:
            self.request_errors += 1
    
    def record_quantization(self, duration: float, error: bool = False) -> None:
        """Record quantization metrics."""
        self.quantization_count += 1
        self.quantization_duration_sum += duration
        if error:
            self.quantization_errors += 1
    
    def record_compression(self, original_bytes: int, compressed_bytes: int) -> None:
        """Record compression metrics."""
        self.total_bytes_original += original_bytes
        self.total_bytes_compressed += compressed_bytes
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics in Prometheus format.
        
        Returns:
            Dict[str, Any]: Metrics dictionary
        """
        uptime = time.time() - self.start_time
        
        # Calculate rates and averages
        avg_request_duration = (
            self.request_duration_sum / self.request_count 
            if self.request_count > 0 else 0.0
        )
        avg_quantization_duration = (
            self.quantization_duration_sum / self.quantization_count
            if self.quantization_count > 0 else 0.0
        )
        compression_ratio = (
            self.total_bytes_original / self.total_bytes_compressed
            if self.total_bytes_compressed > 0 else 0.0
        )
        
        return {
            # Request metrics
            "llm_compression_requests_total": self.request_count,
            "llm_compression_request_duration_seconds": avg_request_duration,
            "llm_compression_request_errors_total": self.request_errors,
            
            # Quantization metrics
            "llm_compression_quantizations_total": self.quantization_count,
            "llm_compression_quantization_duration_seconds": avg_quantization_duration,
            "llm_compression_quantization_errors_total": self.quantization_errors,
            
            # Compression metrics
            "llm_compression_bytes_original_total": self.total_bytes_original,
            "llm_compression_bytes_compressed_total": self.total_bytes_compressed,
            "llm_compression_ratio": compression_ratio,
            
            # System metrics
            "llm_compression_uptime_seconds": uptime,
        }
    
    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus text format.
        
        Returns:
            str: Prometheus formatted metrics
        """
        metrics = self.get_metrics()
        lines = []
        
        for name, value in metrics.items():
            # Add HELP and TYPE
            lines.append(f"# HELP {name} {name.replace('_', ' ').title()}")
            metric_type = "counter" if "total" in name else "gauge"
            lines.append(f"# TYPE {name} {metric_type}")
            lines.append(f"{name} {value}")
            lines.append("")
        
        return "\n".join(lines)


# Global metrics collector instance
_metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    return _metrics_collector
