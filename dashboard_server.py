"""
AI-OS Visual Cortex (Dashboard Server).

A zero-dependency HTTP server that provides:
1. Static file serving (Cyberpunk UI).
2. JSON API endpoints for system status.
3. Server-Sent Events (SSE) for real-time thought streaming.

Usage:
    server = DashboardServer(arrow_engine, factory, port=8000)
    server.start()
"""

import json
import time
import logging
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)

# Shared Usage State
class DashboardData:
    logs = []
    engine = None
    factory = None
    lock = threading.Lock()
    
    next_id = 0
    
    @classmethod
    def append_log(cls, msg: str, level: str = "INFO"):
        with cls.lock:
            entry = {
                "id": cls.next_id,
                "time": time.time(), 
                "msg": msg, 
                "level": level
            }
            cls.logs.append(entry)
            cls.next_id += 1
            if len(cls.logs) > 100:
                cls.logs.pop(0)

    @classmethod
    def get_logs_since(cls, last_id: int) -> List[dict]:
        with cls.lock:
            if not cls.logs:
                return []
            
            # Optimization: check if we are too far behind
            if last_id < cls.logs[0]["id"] - 1:
                return list(cls.logs) # Return all if we missed too much
                
            # Filter
            return [l for l in cls.logs if l["id"] > last_id]

# HTML Template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI-OS Neural Dashboard</title>
    <style>
        :root {
            --bg: #0a0a0a;
            --text: #e0e0e0;
            --accent: #00ff9d;
            --accent-dim: #004d2e;
            --warn: #ffcc00;
            --danger: #ff4444;
            --card-bg: #1a1a1a;
            --border: #333;
        }
        body {
            background: var(--bg);
            color: var(--text);
            font-family: 'Courier New', monospace;
            margin: 0;
            padding: 20px;
            display: grid;
            grid-template-columns: 350px 1fr;
            grid-template-rows: auto 1fr;
            gap: 20px;
            height: 100vh;
            box-sizing: border-box;
            overflow: hidden;
        }
        header {
            grid-column: 1 / -1;
            border-bottom: 2px solid var(--accent);
            padding-bottom: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        h1 { margin: 0; color: var(--accent); text-transform: uppercase; letter-spacing: 2px; font-size: 1.5em; }
        
        .card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            padding: 15px;
            border-radius: 4px;
            display: flex;
            flex-direction: column;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .card h2 {
            margin-top: 0;
            border-bottom: 1px solid var(--border);
            padding-bottom: 8px;
            font-size: 1em;
            color: var(--accent);
            text-transform: uppercase;
        }
        
        #stats-panel { grid-row: 2; display: flex; flex-direction: column; gap: 20px; overflow-y: auto; }
        .stat-group { margin-bottom: 15px; }
        .stat-item { display: flex; justify-content: space-between; margin-bottom: 5px; font-size: 0.9em; }
        .stat-val { font-weight: bold; color: var(--warn); }
        .stat-label { color: #888; }
        
        #main-view { grid-row: 2; display: grid; grid-template-rows: 2fr 1fr; gap: 20px; overflow: hidden; }
        
        #terminal {
            font-family: 'Consolas', monospace;
            font-size: 0.85em;
            flex-grow: 1;
            overflow-y: auto;
            background: #000;
            border: 1px solid var(--accent-dim);
            padding: 10px;
            color: #ccc;
            white-space: pre-wrap;
        }
        .log-entry { margin-bottom: 2px; line-height: 1.4; border-left: 2px solid transparent; padding-left: 5px; }
        .log-info { border-color: #444; }
        .log-warning { color: var(--warn); border-color: var(--warn); }
        .log-error { color: var(--danger); border-color: var(--danger); }
        
        #factory-queue { overflow-y: auto; }
        table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
        th, td { text-align: left; padding: 6px; border-bottom: 1px solid #333; }
        th { color: var(--accent); position: sticky; top: 0; background: var(--card-bg); }
        .status-running { color: var(--warn); animation: pulse 1s infinite; }
        .status-completed { color: var(--accent); }
        .status-failed { color: var(--danger); }
        .status-pending { color: #888; }
        
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
        
        /* Scrollbar */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #111; }
        ::-webkit-scrollbar-thumb { background: #333; }
        ::-webkit-scrollbar-thumb:hover { background: #555; }
    </style>
</head>
<body>
    <header>
        <h1>AI-OS <span style="font-size:0.6em; opacity:0.8">// Visual Cortex v10.0</span></h1>
        <div id="connection-status" style="color:var(--danger)">OFFLINE</div>
    </header>
    
    <div id="stats-panel">
        <div class="card">
            <h2>Cortex Status</h2>
            <div class="stat-group">
                <div class="stat-item"><span class="stat-label">Model:</span><span id="model-name" class="stat-val">...</span></div>
                <div class="stat-item"><span class="stat-label">Device:</span><span id="device-type" class="stat-val">...</span></div>
                <div class="stat-item"><span class="stat-label">Active LoRA:</span><span id="active-lora" class="stat-val">None</span></div>
                <div class="stat-item"><span class="stat-label">Router Conf:</span><span id="router-conf" class="stat-val">0.00</span></div>
            </div>
            <div class="stat-group">
                <div class="stat-item"><span class="stat-label">System Memory:</span><span id="mem-usage" class="stat-val">...</span></div>
                <div class="stat-item"><span class="stat-label">Uptime:</span><span id="uptime" class="stat-val">...</span></div>
            </div>
        </div>
        
        <div class="card">
            <h2>Skill Factory</h2>
            <div class="stat-group">
                <div class="stat-item"><span class="stat-label">State:</span><span id="auto-state" class="stat-val">IDLE</span></div>
                <div class="stat-item"><span class="stat-label">Pending Tasks:</span><span id="pending-tasks" class="stat-val">0</span></div>
            </div>
        </div>
    </div>
    
    <div id="main-view">
        <div class="card">
            <h2>Thought Stream</h2>
            <div id="terminal"></div>
        </div>
        
        <div class="card">
            <h2>Production Queue</h2>
            <div id="factory-queue">
                <table>
                    <thead><tr><th>ID</th><th>Task</th><th>Type</th><th>Status</th></tr></thead>
                    <tbody id="queue-body"></tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        const statusEl = document.getElementById('connection-status');
        const termEl = document.getElementById('terminal');
        const queueEl = document.getElementById('queue-body');
        
        // SSE for Logs
        const evtSource = new EventSource("/stream");
        
        evtSource.onopen = () => {
            statusEl.textContent = "SYSTEM ONLINE";
            statusEl.style.color = "var(--accent)";
            log("Visual Cortex connected to Brain.", "INFO");
        };
        
        evtSource.onerror = () => {
            statusEl.textContent = "SIGNAL LOST";
            statusEl.style.color = "var(--danger)";
        };
        
        evtSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'log') {
                    log(data.msg, data.level);
                }
            } catch(e) { console.error(e); }
        };

        function log(msg, level='INFO') {
            const div = document.createElement('div');
            div.className = `log-entry log-${level.toLowerCase()}`;
            // Format time
            const time = new Date().toLocaleTimeString();
            div.innerHTML = `<span style="opacity:0.5">[${time}]</span> ${msg}`;
            termEl.appendChild(div);
            // Auto scroll if near bottom
            if(termEl.scrollHeight - termEl.scrollTop - termEl.clientHeight < 100) {
                termEl.scrollTop = termEl.scrollHeight;
            }
        }

        function updateStats(stats) {
            document.getElementById('model-name').textContent = stats.model || 'Unknown';
            document.getElementById('device-type').textContent = stats.device || 'CPU';
            document.getElementById('active-lora').textContent = stats.top_lora || 'None';
            document.getElementById('router-conf').textContent = (stats.confidence || 0).toFixed(2);
            document.getElementById('mem-usage').textContent = stats.memory || '---';
            document.getElementById('uptime').textContent = stats.uptime || '0s';
            
            document.getElementById('auto-state').textContent = stats.factory_busy ? 'PROCESSING' : 'IDLE';
            document.getElementById('pending-tasks').textContent = stats.pending_count || 0;
            
            // Colorize state
            const stateEl = document.getElementById('auto-state');
            if (stats.factory_busy) {
                stateEl.style.color = 'var(--warn)';
                stateEl.classList.add('pulse');
            } else {
                stateEl.style.color = 'var(--accent)';
                stateEl.classList.remove('pulse');
            }
            
            // Update Queue
            if (stats.queue) {
                queueEl.innerHTML = stats.queue.map(t => `
                    <tr>
                        <td style="font-family:monospace; opacity:0.7">${t.id.substring(0,6)}</td>
                        <td>${t.name}</td>
                        <td style="font-size:0.8em">${t.type}</td>
                        <td class="status-${t.status}">${t.status.toUpperCase()}</td>
                    </tr>
                `).join('');
            }
        }
        
        // Polling loop
        setInterval(async () => {
            try {
                const res = await fetch('/api/status');
                if(res.ok) {
                    const data = await res.json();
                    updateStats(data);
                }
            } catch(e) { console.log("Poll failed", e); }
        }, 1000);
    </script>
</body>
</html>
"""

class APIHandler(BaseHTTPRequestHandler):
    """Custom handler for API endoints."""
    
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))
            return
            
        if self.path == "/api/status":
            self._handle_status()
            return
            
        if self.path == "/stream":
            self._handle_stream()
            return
            
        # Default (404)
        self.send_error(404)

    def _handle_status(self):
        """Return JSON status of Engine + Factory."""
        try:
            import psutil
            mem = f"{psutil.Process().memory_info().rss / 1024 / 1024:.0f} MB"
        except ImportError:
            mem = "N/A"
        
        status = {
            "model": "Unknown",
            "device": "Unknown",
            "memory": mem,
            "uptime": f"{time.time() - self.server.start_time:.0f}s",
            "factory_busy": False,
            "pending_count": 0,
            "queue": [],
            "top_lora": "None",
            "confidence": 0.0
        }
        
        # Populate from Engine
        if DashboardData.engine:
            status["model"] = DashboardData.engine.metadata.get("model_name", "MiniLM")
            status["device"] = getattr(DashboardData.engine, "device", "cpu")
            status["top_lora"] = getattr(DashboardData.engine, "last_used_lora", "None") or "None"
            status["confidence"] = getattr(DashboardData.engine, "last_router_confidence", 0.0)
        # Populate from Factory
        if DashboardData.factory:
            tasks = list(DashboardData.factory.tasks.values())
            tasks.sort(key=lambda t: t.created_at, reverse=True)
            status["queue"] = [
                {"id": t.id, "name": t.name, "type": t.type, "status": t.status}
                for t in tasks[:10]
            ]
            status["factory_busy"] = any(t.status == "running" for t in tasks)
            status["pending_count"] = sum(1 for t in tasks if t.status == "pending")

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(status).encode("utf-8"))

    def _handle_stream(self):
        """SSE Stream implementation."""
        self.send_response(200)
        self.send_header("Content-type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        
        last_id = -1
        
        try:
            while True:
                # Fetch new logs by ID
                new_logs = DashboardData.get_logs_since(last_id)
                
                if new_logs:
                    for log in new_logs:
                        data = json.dumps({"type": "log", "msg": log["msg"], "level": log["level"]})
                        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                        last_id = log["id"]
                    self.wfile.flush()
                
                time.sleep(0.5)
        except (BrokenPipeError, ConnectionAbortedError):
            pass
            
    def log_message(self, format, *args):
        pass



class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

class DashboardServer:
    """Wrapper to run server in background."""
    
    def __init__(self, engine, factory, port=8000):
        # Update references (Always needed for fresh engine instance)
        DashboardData.engine = engine
        DashboardData.factory = factory
        
        # Setup logging interception (Only once)
        root = logging.getLogger()
        has_dash_handler = any(isinstance(h, DashboardLogHandler) for h in root.handlers)
        
        if not has_dash_handler:
            handler = DashboardLogHandler()
            handler.setLevel(logging.INFO)
            root.addHandler(handler)
            
        self.port = port
        self.httpd = ReusableThreadingHTTPServer(("0.0.0.0", port), APIHandler)
        self.httpd.start_time = time.time()
        self.thread = None
        
    def start(self):
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        logger.info(f"Visual Cortex (Dashboard) online at http://localhost:{self.port}")
        
    def stop(self):
        self.httpd.shutdown()


class DashboardLogHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            DashboardData.append_log(msg, record.levelname)
        except Exception:
            pass
