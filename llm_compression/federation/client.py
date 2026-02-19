
"""
AI-OS Arrow Flight Federation Client.

Connects to remote AI-OS nodes to discover and download skills.
"""

import pyarrow.flight as flight
import pyarrow as pa
from typing import List, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class LoRAFlightClient:
    """Client for interacting with LoRAFlightServer peers."""
    
    def __init__(self, peer_uri: str):
        """
        Connect to a peer.
        Args:
            peer_uri: e.g. "grpc://192.168.1.5:8888"
        """
        self.peer_uri = peer_uri
        try:
            self.client = flight.FlightClient(peer_uri)
            logger.info(f"Connected to Flight Server at {peer_uri}")
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise

    def list_skills(self) -> List[Dict[str, Any]]:
        """List available LoRAs on remote peer."""
        skills = []
        try:
            for info in self.client.list_flights():
                desc = info.descriptor
                endpoint = info.endpoints[0]
                # key is usually the filename or identifier
                key = desc.path[0].decode('utf-8')
                
                skills.append({
                    "name": key,
                    "size_bytes": info.total_bytes,
                    # We could also parse schema metadata if we implemented it in list_flights
                })
        except Exception as e:
            logger.error(f"Failed to list skills from {self.peer_uri}: {e}")
            raise
        return skills

    def fetch_skill(self, skill_name: str, save_path: str, overwrite: bool = False):
        """
        Download a LoRA skill to local path.
        
        Args:
            skill_name: The identifier of the LoRA (filename).
            save_path: Local path to save .lora.arrow file.
            overwrite: Whether to overwrite existing file.
        """
        path = Path(save_path)
        
        if path.exists() and not overwrite:
            logger.info(f"Skill {skill_name} already exists at {path}. Skipping.")
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        
        ticket = flight.Ticket(skill_name.encode('utf-8'))
        
        logger.info(f"Downloading {skill_name} from {self.peer_uri}...")
        
        try:
            reader = self.client.do_get(ticket)
            
            # Use the schema from the stream which includes metadata
            schema = reader.schema
            
            # Write to disk using Arrow IPC
            with pa.OSFile(str(path), 'wb') as sink:
                with pa.ipc.new_file(sink, schema) as writer:
                    total_rows = 0
                    for chunk in reader:
                         # chunk is FlightStreamChunk
                         if chunk.data:
                             writer.write(chunk.data)
                             total_rows += chunk.data.num_rows
                         
            logger.info(f"Downloaded {skill_name} ({total_rows} rows) to {path}")
            
        except Exception as e:
            logger.error(f"Failed to fetch {skill_name}: {e}")
            if path.exists():
                path.unlink() # Cleanup partial file
            raise
