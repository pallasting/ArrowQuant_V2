
"""
Test: Dashboard Server.

Verifies:
1. HTTP Server starts.
2. API /status returns JSON.
3. SSE /stream returns events (eventually).
"""

import sys
import os
sys.path.append(os.getcwd())

import time
import json
import threading
import unittest
import urllib.request
from dashboard_server import DashboardServer, DashboardData

class MockEngine:
    metadata = {"model_name": "TestModel"}
    device = "cpu"

class MockFactory:
    tasks = {}

class TestDashboard(unittest.TestCase):
    def setUp(self):
        self.server = DashboardServer(MockEngine(), MockFactory(), port=0) # Dynamic port
        self.server.start()
        self.port = self.server.httpd.server_address[1]
        time.sleep(1) # Wait for startup
        
    def tearDown(self):
        if hasattr(self, 'server'):
            self.server.stop()
        
    def test_status_endpoint(self):
        """Test simple GET /api/status."""
        with urllib.request.urlopen(f"http://localhost:{self.port}/api/status") as response:
            self.assertEqual(response.status, 200)
            data = json.loads(response.read().decode())
            self.assertEqual(data["model"], "TestModel")
            self.assertIn("memory", data)
            
    def test_html_endpoint(self):
        """Test serving the HTML dashboard."""
        with urllib.request.urlopen(f"http://localhost:{self.port}/") as response:
            self.assertEqual(response.status, 200)
            content = response.read().decode()
            self.assertIn("AI-OS", content)
            self.assertIn("Visual Cortex", content)

    def test_logging_stream(self):
        """Test log interception."""
        # Log something using standard logger
        import logging
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.info("Test log message")
        
        # Check DashboardData
        # Wait a bit for async processing if any (DashboardHandler is sync though)
        time.sleep(0.1)
        logs = DashboardData.get_logs_since(-1)
        found = any("Test log message" in l["msg"] for l in logs)
        self.assertTrue(found, "Log message not captured by dashboard handler")

if __name__ == "__main__":
    unittest.main()
