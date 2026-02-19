# AI-OS 2.0 Deployment Checklist

**Status**: Ready for Production (User Verification Required)
**Version**: 2.0.0 (RC1)

This checklist ensures your AI-OS environment is fully configured and ready for autonomous operation.

## 1. Environment Setup
- [ ] **Python 3.10+** (Recommend 3.11 for performance).
- [ ] **Dependencies**: `pip install -r requirements.txt`.
- [ ] **Hardware Acceleration**:
  - [ ] NVIDIA GPU (CUDA via `torch`) OR
  - [ ] Intel CPU (AVX512 optimizations enabled).
- [ ] **Ollama**: Installed and running (`ollama serve`).

## 2. Model Initialization
- [ ] **Embedding Model**: Ensure `models/minilm` exists (or similar).
  - [ ] Download script: `python scripts/download_model.py` (if available) or manual download.
- [ ] **Vision Model**: OpenCV ready (check with `python scripts/check_sensors.py`).
- [ ] **Audio Model**: SoundDevice ready.

## 3. Sensor Calibration
- [ ] **Camera**: Visible and clear image.
  - Test: `python scripts/check_sensors.py --vision`
- [ ] **Microphone**: Audio levels detected.
  - Test: `python scripts/check_sensors.py --audio`

## 4. Action Verification (Phase 12)
- [ ] **Demonstration Trace**: Record a simple action (move mouse).
  - `python scripts/record_demo.py`
- [ ] **Replay Verification**: Replay the action.
  - `python scripts/replay_demo.py`
  - *Must execute without infinite loop (fixed in Phase 12.4).*

## 5. Agent Activation (Phase 13)
- [ ] **Basic Cognition**:
  - `python scripts/run_agent.py "Wait for 2 seconds"`
  - *Success Criteria*: Connects to Ollama, receives "wait" decision, executes "wait".
- [ ] **Full Autonomy**:
  - `python scripts/run_agent.py "Open Calculator"`
  - *Success Criteria*: Observes screen, finding Calc icon, clicks it.

## 6. System Monitoring (Phase 10)
- [ ] **Dashboard**:
  - `python dashboard_server.py`
  - Open `http://localhost:8000/dashboard.html`
  - *Verify live sensor log and memory updates.*

## 7. Evolution Pipeline (Phase 9)
- [ ] **LoRA Training**:
  - Verify `lora_skills/` directory is writable.
  - *Optional*: Run `python scripts/demo_lora_system.py` to test skill extraction.

---

**Troubleshooting:**
- **"Connection Refused"**: Check Ollama (`netstat -an | findstr 11434`).
- **"Permission Denied"**: Run terminal as Administrator (for ActionManager).
- **"Camera index out of range"**: Check `cv2.VideoCapture(0)` index.
