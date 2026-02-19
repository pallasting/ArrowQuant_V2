# AI-OS: Self-Evolving Intelligent Memory System

**Version**: 2.0 (Arrow-Native + Embodied Action)
**Status**: Stable / Feature Complete

AI-OS is a next-generation operating system memory layer that evolves with the user. It combines high-performance vector retrieval, multimodal perception, and embodied action capabilities.

## 🚀 Key Features

### 1. **Arrow-Native Memory Engine** (Phase 2-6)
- **Zero-Copy Retrieval**: Powered by Apache Arrow and PyArrow.
- **Lightning Fast**: <5ms inference latency, >2000 queries/sec throughput.
- **Hybrid Storage**: Hierarchical storage of raw data (Parquet) and embeddings (Faiss/Arrow).

### 2. **Self-Evolving Intelligence** (Phase 7-9)
- **Fluid Intelligence (LoRA)**: Adapts to new tasks by learning lightweight "Skill Cards".
- **Federated Learning**: Share skills across devices via decentralized Flight RPC.
- **Cognitive Dissonance**: Automatically triggers learning when the model is confused.

### 3. **Multimodal Perception** (Phase 10-11)
- **Vision**: "Eyes" that see the desktop and webcam (OpenCV).
- **Audio**: "Ears" that listen and process command (SoundDevice).
- **Proprioception**: Self-awareness of CPU/RAM/Battery state.

### 4. **Embodied Action** (Phase 12-13)
- **Autonomous Agent**: Integrated OODA Loop (Observe-Orient-Decide-Act).
- **"Watch & Do"**: Learn by watching user demonstrations (Imitation Learning).
- **Safety Sandbox**: Restricted action space to prevent system damage.

---

## 📦 Quick Start

### 1. Prerequisites
- Python 3.10+
- Recommended: NVIDIA GPU (CUDA) or Intel CPU (AVX512) for acceleration.
- Ollama (for LLM reasoning): `ollama serve`

### 2. Installation
```bash
# Clone and enter
git clone https://github.com/ai-os/memory.git
cd ai-os-memory

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Autonomous Agent
Launch the agent with a natural language goal:
```bash
python scripts/run_agent.py "Wait for 10 seconds"
```

To enable Vision (Webcam):
```bash
python scripts/run_agent.py "Describe what you see" --vision
```

### 4. Run the Dashboard
Visualize the system's "thought process":
```bash
python dashboard_server.py
# Open http://localhost:8000/dashboard.html
```

---

## 📚 Documentation

### Core Guides
- [**Quick Start Guide**](QUICK_START_GUIDE.md): Detailed setup instructions.
- [**Architecture Overview**](LORA_ARCHITECTURE_AND_STATUS.md): Deep dive into the system design.
- [**Agent Usage**](AGENT_USAGE.md): How to use the Autonomous Agent.

### Developer API
- [**ArrowEngine API**](docs/ARROW_API_REFERENCE.md): Vector database and inference.
- [**Multimodal API**](docs/MULTIMODAL_COMPLETE_GUIDE.md): Vision and Audio sensors.

### Status Reports
- [**Phase 1-2**](docs/PHASE_2.0_PERFORMANCE_BENCHMARK_REPORT.md): Initial optimizations.
- [**Phase 12**](docs/TASK_12_COMPLETION_FINAL.md): Embodied Action implementation.

---

## 🛠 Project Structure

```
ai-os-memory/
├── llm_compression/
│   ├── agent/           # Autonomous Agent (Phase 13)
│   ├── action/          # Action Manager (Phase 12)
│   ├── evolution/       # Self-Evolution / LoRA (Phase 9)
│   ├── federation/      # Decentralized Sharing (Phase 8)
│   ├── inference/       # ArrowEngine Core (Phase 2-7)
│   ├── multimodal/      # Vision/Audio Processing (Phase 11)
│   └── sensors/         # Hardware Sensors
├── scripts/             # Utility scripts (run_agent, record_demo, etc.)
├── docs/                # Comprehensive documentation
├── tests/               # Unit and Integration tests
└── config/              # Configuration files
```

## 🤝 Contributing
Contributions are welcome! Please read `docs/CONTRIBUTING.md` (Coming Soon).

## 📄 License
MIT License
