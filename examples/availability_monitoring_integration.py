"""
Availability Monitoring Integration Example

Demonstrates how to integrate availability monitoring into the LLM compression system
to achieve >99.9% availability.
"""

import asyncio
from pathlib import Path

from llm_compression.availability_monitor import (
    AvailabilityMonitor,
    ComponentType
)
from llm_compression.circuit_breaker import (
    CircuitBreakerRegistry,
    CircuitBreakerConfig
)
from llm_compression.health_endpoints import HealthEndpoints


class SystemWithAvailabilityMonitoring:
    """
    Example system with integrated availability monitoring
    """
    
    def __init__(self):
        # Initialize monitoring components
        self.availability_monitor = AvailabilityMonitor(
            check_interval=60.0,  # Check every minute
            history_file=Path("availability_history.jsonl")
        )
        
        self.circuit_breakers = CircuitBreakerRegistry()
        
        self.health_endpoints = HealthEndpoints(
            self.availability_monitor,
            self.circuit_breakers
        )
        
        # Configure circuit breakers for critical components
        self.storage_breaker = self.circuit_breakers.get_or_create(
            "storage",
            CircuitBreakerConfig(
                failure_threshold=5,
                success_threshold=2,
                timeout=30.0
            )
        )
        
        self.embedder_breaker = self.circuit_breakers.get_or_create(
            "embedder",
            CircuitBreakerConfig(
                failure_threshold=3,
                success_threshold=2,
                timeout=60.0
            )
        )
    
    async def start(self):
        """Start the system with monitoring"""
        print("Starting availability monitoring...")
        await self.availability_monitor.start()
        
        # Start periodic health checks
        asyncio.create_task(self._periodic_health_checks())
        
        print("System started with availability monitoring")
    
    async def stop(self):
        """Stop the system"""
        print("Stopping availability monitoring...")
        await self.availability_monitor.stop()
        print("System stopped")
    
    async def _periodic_health_checks(self):
        """Perform periodic health checks on all components"""
        while True:
            try:
                # Check storage component
                await self.availability_monitor.check_health(
                    ComponentType.STORAGE.value,
                    self._check_storage_health
                )
                
                # Check embedder component
                await self.availability_monitor.check_health(
                    ComponentType.EMBEDDER.value,
                    self._check_embedder_health
                )
                
                # Check vector search component
                await self.availability_monitor.check_health(
                    ComponentType.VECTOR_SEARCH.value,
                    self._check_vector_search_health
                )
                
                # Check semantic indexer
                await self.availability_monitor.check_health(
                    ComponentType.SEMANTIC_INDEXER.value,
                    self._check_semantic_indexer_health
                )
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Health check error: {e}")
    
    async def _check_storage_health(self):
        """Health check for storage component"""
        # Simulate storage health check
        # In real implementation, this would check Arrow storage
        await asyncio.sleep(0.01)
        return True
    
    async def _check_embedder_health(self):
        """Health check for embedder component"""
        # Simulate embedder health check
        # In real implementation, this would test embedding generation
        await asyncio.sleep(0.02)
        return True
    
    async def _check_vector_search_health(self):
        """Health check for vector search component"""
        # Simulate vector search health check
        # In real implementation, this would test search functionality
        await asyncio.sleep(0.01)
        return True
    
    async def _check_semantic_indexer_health(self):
        """Health check for semantic indexer component"""
        # Simulate semantic indexer health check
        # In real implementation, this would test LLM indexing
        await asyncio.sleep(0.03)
        return True
    
    async def compress_with_circuit_breaker(self, text: str):
        """
        Compress text with circuit breaker protection
        
        Args:
            text: Text to compress
            
        Returns:
            Compressed result
        """
        # Use circuit breaker for storage operations
        async def storage_operation():
            # Simulate compression
            await asyncio.sleep(0.1)
            return {"compressed": True, "size": len(text)}
        
        result = await self.storage_breaker.call(storage_operation)
        return result
    
    async def get_system_status(self):
        """Get comprehensive system status"""
        # Get health status
        health = await self.health_endpoints.health()
        
        # Get availability metrics
        metrics = await self.health_endpoints.metrics()
        
        # Get readiness
        readiness = await self.health_endpoints.readiness()
        
        # Get liveness
        liveness = await self.health_endpoints.liveness()
        
        return {
            'health': health,
            'metrics': metrics,
            'readiness': readiness,
            'liveness': liveness
        }


async def main():
    """Main example"""
    print("=== Availability Monitoring Integration Example ===\n")
    
    # Create system with monitoring
    system = SystemWithAvailabilityMonitoring()
    
    # Start system
    await system.start()
    
    # Simulate some operations
    print("\nSimulating system operations...")
    for i in range(5):
        try:
            result = await system.compress_with_circuit_breaker(f"Test text {i}")
            print(f"Operation {i+1}: Success - {result}")
        except Exception as e:
            print(f"Operation {i+1}: Failed - {e}")
        
        await asyncio.sleep(1)
    
    # Wait for health checks
    print("\nWaiting for health checks...")
    await asyncio.sleep(5)
    
    # Get system status
    print("\n=== System Status ===")
    status = await system.get_system_status()
    
    print(f"\nHealth Status: {status['health']['status']}")
    print(f"Availability: {status['health']['availability_percentage']:.2f}%")
    print(f"Uptime: {status['health']['uptime_seconds']:.1f}s")
    print(f"Active Incidents: {status['health']['active_incidents']}")
    
    print(f"\nReadiness: {'Ready' if status['readiness']['ready'] else 'Not Ready'}")
    print(f"Liveness: {'Alive' if status['liveness']['alive'] else 'Dead'}")
    
    print(f"\nHealth Checks:")
    print(f"  Total: {status['metrics']['health_checks']['total']}")
    print(f"  Successful: {status['metrics']['health_checks']['successful']}")
    print(f"  Failed: {status['metrics']['health_checks']['failed']}")
    print(f"  Success Rate: {status['metrics']['health_checks']['success_rate']:.1f}%")
    
    print(f"\nAvailability Metrics (24h):")
    metrics_24h = status['metrics']['availability']['24h']
    print(f"  Availability: {metrics_24h['percentage']:.4f}%")
    print(f"  Uptime: {metrics_24h['uptime_seconds']:.1f}s")
    print(f"  Downtime: {metrics_24h['downtime_seconds']:.1f}s")
    print(f"  MTBF: {metrics_24h['mtbf']:.1f}s")
    print(f"  MTTR: {metrics_24h['mttr']:.1f}s")
    
    # Check if we meet 99.9% availability target
    availability = status['health']['availability_percentage']
    if availability >= 99.9:
        print(f"\n✅ SUCCESS: Availability {availability:.4f}% meets >99.9% target!")
    else:
        print(f"\n⚠️  WARNING: Availability {availability:.4f}% below 99.9% target")
    
    # Stop system
    await system.stop()
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
