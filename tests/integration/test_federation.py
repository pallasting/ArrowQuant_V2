
import threading
import time
import shutil
import unittest
import pyarrow.flight as flight
from pathlib import Path
import numpy as np
import logging

# Set up logging for test
logging.basicConfig(level=logging.INFO)

from llm_compression.federation.server import LoRAFlightServer
from llm_compression.federation.client import LoRAFlightClient
from llm_compression.inference.lora_format import LoRACard, LoRAFormat

class TestFederation(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_federation_workspace")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            
        self.server_dir = self.test_dir / "server_storage"
        self.client_dir = self.test_dir / "client_storage"
        
        self.server_dir.mkdir(parents=True)
        self.client_dir.mkdir(parents=True)
        
        # Create Dummy LoRA
        self.card = LoRACard(
            name="shared_skill",
            rank=4,
            alpha=16.0,
            target_modules=["query"],
            weights_A={"l1.query": np.zeros((4, 32)).astype(np.float32)},
            weights_B={"l1.query": np.zeros((32, 4)).astype(np.float32)},
            metadata={"description": "Test Skill for Federation"}
        )
        LoRAFormat.save(self.card, str(self.server_dir / "shared_skill.lora.arrow"))

        # Find free port? Or use fixed port for test
        self.port = 19090
        self.location = f"grpc://localhost:{self.port}"
        
        self.server = LoRAFlightServer(self.location, str(self.server_dir))
        
        # Run server in thread
        # print(dir(self.server)) # Debug if needed
        # Use wait() if serve() is missing?
        self.server_thread = threading.Thread(target=self.server.wait)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        # Give it a moment to bind
        time.sleep(1)

    def tearDown(self):
        try:
            self.server.shutdown()
            self.server_thread.join(timeout=2)
        except Exception as e:
            print(f"Error shutting down server: {e}")
            
        if self.test_dir.exists():
            try:
                shutil.rmtree(self.test_dir)
            except:
                pass

    def test_list_and_download(self):
        """Test listing and downloading a LoRA skill over Arrow Flight."""
        
        # Connect Client
        client = LoRAFlightClient(self.location)
        
        # 1. List
        skills = client.list_skills()
        print(f"Discovered skills: {skills}")
        
        # Depending on flight implementation, 'name' might be path relative to root
        # In server.py we yielded `path.name`
        
        found = False
        for s in skills:
            if s['name'] == "shared_skill.lora.arrow":
                found = True
                break
        self.assertTrue(found, "shared_skill.lora.arrow not found in listing")
        
        # 2. Download
        save_path = self.client_dir / "downloaded_skill.lora.arrow"
        client.fetch_skill("shared_skill.lora.arrow", str(save_path))
        
        self.assertTrue(save_path.exists(), "Downloaded file does not exist")
        
        # 3. Verify content
        loaded_card = LoRAFormat.load(str(save_path))
        self.assertEqual(loaded_card.name, "shared_skill")
        self.assertEqual(loaded_card.metadata["description"], "Test Skill for Federation")
        self.assertTrue("l1.query" in loaded_card.weights_A)
        print("Federation test passed!")

if __name__ == "__main__":
    unittest.main()
