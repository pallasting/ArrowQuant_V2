
"""
AI-OS Federation Manager.

Orchestrates the lifecycle of the Federated Network:
1. Starts the local Flight Server (Sharing).
2. Starts the Discovery Service (Finding Peers).
3. Manages connections and skill indexing from the swarm.
"""

import threading
import time
import logging
from typing import Dict, List, Optional
from pathlib import Path
from .discovery import DiscoveryService
from .client import LoRAFlightClient
from .server import LoRAFlightServer

logger = logging.getLogger(__name__)

class FederationManager:
    """
    Central hub for AI-OS Federation.
    """
    def __init__(self, node_name: str, port: int, storage_dir: str):
        self.node_name = node_name
        self.port = port
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Server (Serving local storage_dir)
        # Location uses 0.0.0.0 to bind all interfaces
        self.location = f"grpc://0.0.0.0:{port}"
        self.server = LoRAFlightServer(self.location, str(self.storage_dir))
        
        # Initialize Discovery (Advertising port)
        self.discovery = DiscoveryService(node_name, port)
        
        # State
        self.remote_skills_cache: Dict[str, Dict] = {} # {skill_name: {peer_uri, peer_name, ...}}
        
        # Threads
        self.server_thread = None
        self._running = False
        
    def start(self):
        """Start Federation services."""
        if self._running:
            return
            
        logger.info(f"Starting Federation Node '{self.node_name}' on port {self.port}...")
        
        # 1. Start Flight Server
        # Using wait() based on test findings for this environment
        self.server_thread = threading.Thread(target=self.server.wait, daemon=True)
        self.server_thread.start()
        
        # 2. Start Discovery
        self.discovery.start()
        
        self._running = True
        logger.info("Federation Services Active.")
        
    def stop(self):
        """Stop Federation services."""
        if not self._running:
            return
            
        logger.info("Stopping Federation Services...")
        self.discovery.stop()
        self.server.shutdown()
        
        if self.server_thread and self.server_thread.is_alive():
            # Wait briefly but don't hang
            self.server_thread.join(timeout=1.0)
            
        self._running = False
        logger.info("Federation Stopped.")

    def scan_remote_skills(self) -> Dict[str, Dict]:
        """
        Polls all discovered peers for their available skills.
        Updates internal cache.
        Returns: Dict of new skills found.
        """
        peers = self.discovery.get_peers()
        logger.info(f"Scanning {len(peers)} peers for skills...")
        
        current_scan = {}
        
        for peer in peers:
            uri = peer['uri']
            name = peer['name']
            
            # Skip self if discovered (shouldn't happen with proper filtering but safe to check)
            # Actually URI might be localhost if running locally
            # We can check node_name if we broadcast it in properties
            
            try:
                # Create ephemeral client
                client = LoRAFlightClient(uri)
                skills = client.list_skills()
                
                for s in skills:
                    # s typically has {'name': 'foo.arrow', 'size_bytes': ...}
                    skill_name = s['name']
                    
                    # Store metadata
                    entry = {
                        "name": skill_name,
                        "peer_uri": uri,
                        "peer_name": name,
                        "size_bytes": s.get('size_bytes', 0),
                        "discovered_at": time.time()
                    }
                    current_scan[skill_name] = entry
                    
            except Exception as e:
                logger.warning(f"Failed to scan peer {name} ({uri}): {e}")
                
        # Update cache
        self.remote_skills_cache.update(current_scan)
        return current_scan

    def fetch_skill(self, skill_name: str, force: bool = False) -> Optional[Path]:
        """
        Downloads a skill from the swarm if available.
        Returns path to local file if successful.
        """
        if skill_name not in self.remote_skills_cache:
            logger.warning(f"Skill '{skill_name}' not found in remote cache.")
            return None
            
        info = self.remote_skills_cache[skill_name]
        uri = info['peer_uri']
        target_path = self.storage_dir / skill_name
        
        if target_path.exists() and not force:
            logger.info(f"Skill '{skill_name}' already exists locally.")
            return target_path
            
        try:
            logger.info(f"Fetching '{skill_name}' from {info['peer_name']}...")
            client = LoRAFlightClient(uri)
            client.fetch_skill(skill_name, str(target_path), overwrite=True)
            return target_path
            
        except Exception as e:
            logger.error(f"Failed to download '{skill_name}': {e}")
            return None
