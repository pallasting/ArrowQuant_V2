
"""
AI-OS Service Discovery Module.

Uses Zeroconf (mDNS) to discover other AI-OS nodes on the local network.
Enables automatic peer finding for the Federation layer.
"""

import socket
import logging
import threading
import time
from typing import Dict, List, Optional, Callable
from zeroconf import ServiceInfo, Zeroconf, ServiceBrowser, ServiceStateChange

logger = logging.getLogger(__name__)

class PeerListener:
    """Listener for Zeroconf service browser."""
    
    def __init__(self, on_change: Optional[Callable] = None):
        self.peers: Dict[str, Dict] = {} # {name: info_dict}
        self.on_change = on_change
        
    def remove_service(self, zeroconf, type, name):
        if name in self.peers:
            logger.info(f"Peer disappeared: {name}")
            del self.peers[name]
            if self.on_change:
                self.on_change("remove", name, None)

    def add_service(self, zeroconf, type, name):
        info = zeroconf.get_service_info(type, name)
        if info:
            self._update_peer(name, info)

    def update_service(self, zeroconf, type, name):
        info = zeroconf.get_service_info(type, name)
        if info:
            self._update_peer(name, info)
            
    def _update_peer(self, name, info: ServiceInfo):
        addresses = [socket.inet_ntoa(addr) for addr in info.addresses]
        if not addresses:
            return
            
        peer_info = {
            "name": name,
            "address": addresses[0],
            "port": info.port,
            "server": info.server,
            "properties": {k.decode('utf-8') if isinstance(k, bytes) else k: 
                           v.decode('utf-8') if isinstance(v, bytes) else v 
                           for k, v in info.properties.items()}
        }
        
        # Build Flight URI
        peer_info["uri"] = f"grpc://{peer_info['address']}:{peer_info['port']}"
        
        self.peers[name] = peer_info
        logger.info(f"Peer discovered: {name} at {peer_info['uri']}")
        
        if self.on_change:
            self.on_change("add", name, peer_info)


class DiscoveryService:
    """
    Manages both advertising this node and discovering other nodes.
    """
    
    SERVICE_TYPE = "_ai-os._tcp.local."
    
    def __init__(self, node_name: str, port: int, properties: dict = None):
        self.zeroconf = Zeroconf()
        self.node_name = node_name
        self.port = port
        self.properties = properties or {}
        
        # Unique service name
        # If node_name collision, zeroconf might handle contentions, but here we assume manual config or UUID
        self.service_name = f"{node_name}.{self.SERVICE_TYPE}"
        self.info = None
        
        self.listener = PeerListener()
        self.browser = None
        
    def start(self):
        """Start advertising and browsing."""
        self._register_service()
        self.browser = ServiceBrowser(self.zeroconf, self.SERVICE_TYPE, self.listener)
        logger.info(f"Discovery Service started via mDNS.")
        
    def stop(self):
        """Stop advertising and browsing."""
        if self.browser:
            self.browser.cancel()
            self.browser = None
            
        if self.info:
            self.zeroconf.unregister_service(self.info)
            self.info = None
            
        self.zeroconf.close()
        logger.info("Discovery Service stopped.")
        
    def get_peers(self) -> List[Dict]:
        """Return list of discovered peers."""
        return list(self.listener.peers.values())
        
    def _register_service(self):
        """Register local service."""
        # Get local IP
        ip = self._get_local_ip()
        
        self.info = ServiceInfo(
            self.SERVICE_TYPE,
            self.service_name,
            addresses=[socket.inet_aton(ip)],
            port=self.port,
            properties=self.properties,
            server=f"{socket.gethostname()}.local."
        )
        
        logger.info(f"Registering {self.service_name} at {ip}:{self.port}")
        try:
            self.zeroconf.register_service(self.info)
        except Exception as e:
            logger.error(f"Failed to register service: {e}")

    def _get_local_ip(self):
        """Best effort local IP."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # doesn't even have to be reachable
            s.connect(('10.255.255.255', 1))
            ip = s.getsockname()[0]
        except Exception:
            ip = '127.0.0.1'
        finally:
            s.close()
        return ip

if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)
    d = DiscoveryService("ai-os-test-node", 8888, {"model": "bert-tiny"})
    d.start()
    try:
        while True:
            time.sleep(5)
            peers = d.get_peers()
            print(f"Peers: {len(peers)}")
    except KeyboardInterrupt:
        d.stop()
