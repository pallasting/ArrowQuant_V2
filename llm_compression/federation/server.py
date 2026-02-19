
"""
AI-OS Arrow Flight Federation Server.

Implements a peer-to-peer LoRA exchange server.
"""

import pyarrow as pa
import pyarrow.flight as flight
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class LoRAFlightServer(flight.FlightServerBase):
    """
    Serves LoRA '.arrow' files from a directory to authorized peers.
    
    Supports:
    - List available LoRAs (list_flights)
    - Get LoRA Metadata (get_flight_info)
    - Download LoRA (do_get)
    """
    
    def __init__(self, location: str, root_dir: str):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.location = location
        super().__init__(location)
        logger.info(f"LoRA Flight Server listening on {location}, serving {root_dir}")

    def list_flights(self, context, criteria):
        """List all .lora.arrow files in the directory."""
        total = 0
        for path in self.root_dir.glob("*.lora.arrow"):
            # Reading every file schema for listing is too slow.
            # Return minimal info for browsing.
            # The name is just the filename.
            
            descriptor = flight.FlightDescriptor.for_path(path.name)
            
            # Simple metadata endpoint
            endpoints = [flight.FlightEndpoint(path.name, [self.location])]
            
            # We don't read schema here to be fast. Assuming client calls get_flight_info later.
            # Or we must provide schema? FlightInfo requires a schema.
            # Use empty schema with special key-value metadata?
            # Or construct minimal schema with just mandatory fields?
            
            # For correctness, let's include basic schema if cached?
            # For now, simplistic approach: empty schema.
            
            schema = pa.schema([]) 
            
            yield flight.FlightInfo(
                 schema,
                 descriptor,
                 endpoints,
                 -1, # total records unknown without reading
                 path.stat().st_size
            )
            total += 1
            
        logger.debug(f"Listed {total} LoRA files.")

    def get_flight_info(self, context, descriptor):
        """Get detailed metadata for a specific LoRA."""
        key = descriptor.path[0].decode('utf-8')
        path = self.root_dir / key
        
        if not path.exists():
            raise flight.FlightUnavailableError(f"LoRA '{key}' not found.")
            
        try:
            with pa.ipc.open_file(path) as reader:
                schema = reader.schema
                descriptor = flight.FlightDescriptor.for_path(key)
                endpoints = [flight.FlightEndpoint(key, [self.location])]
                
                return flight.FlightInfo(
                    schema,
                    descriptor,
                    endpoints,
                    reader.num_record_batches,
                    path.stat().st_size
                )
        except Exception as e:
            logger.error(f"Failed to read metadata for {key}: {e}")
            raise flight.FlightServerError(f"Invalid LoRA file: {key}")

    def do_get(self, context, ticket):
        """Stream the LoRA file content as Arrow RecordBatches."""
        key = ticket.ticket.decode('utf-8')
        path = self.root_dir / key
        
        if not path.exists():
            raise flight.FlightUnavailableError(f"LoRA '{key}' not found.")
            
        logger.info(f"Streaming LoRA: {key}")
        
        try:
            # Use memory mapping for zero-copy read
            source = pa.memory_map(str(path), 'r')
            reader = pa.ipc.open_file(source)
            
            return flight.GeneratorStream(
                reader.schema,
                self._yield_batches_and_close(reader, source)
            )
            
        except Exception as e:
            logger.error(f"Error preparing stream for {key}: {e}")
            raise flight.FlightServerError(f"Streaming error: {e}")

    def _yield_batches_and_close(self, reader, source):
        """Generator that yields batches and ensures source is closed."""
        try:
            for i in range(reader.num_record_batches):
                yield reader.get_batch(i)
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            raise
        finally:
            # Ensure file handle is released
            try:
                source.close()
            except:
                pass
