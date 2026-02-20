# Embodied Voice Agent - System Design

## 1. High-Level Architecture
This component connects the raw auditory sensor stream, local ASR/STT generation, semantic parsing (or Cognitive LLM loop in the future), Action (GUI) generation, and active multimodal expressive behavior (TTS).

```
+---------------+           +---------------+           +-----------------+
| System Mic    |   VAD     | ASREngine     |   Text    | Intent Matcher  |
| (AudioModule) | --------> | (faster_whisper)--------> | (CognitiveLoop) |
+---------------+           +---------------+           +--------+--------+
                                                                 |
                                                                 | intent + args
                                                                 v
+---------------+   Audio   +---------------+           +-----------------+
| System Speaker| <-------- | TTSEngine     |  Narrate  | ActionManager   |
| (sounddevice) |           | (piper-tts)   | <-------- | (PyAutoGUI)     |
+---------------+           +---------------+     |     +-----------------+
                                                  |        | Action Output
                                                  v        v
                                              +---------+ +---------+
                                              | Safety  | | OS GUI  |
                                              | Monitor | | / Screen|
                                              +---------+ +---------+
```

## 2. Core Modules
### 1. `AudioModule` (`llm_compression/sensors/audio.py`)
- Background listener thread using `sounddevice.InputStream`.
- **Latency Optimization:** Blocksize parameter strictly set (e.g. 1024), utilizing `dtype=float32` arrays.
- **Voice Activity Detector (VAD):** Calculates the energy peak (using amplitude metrics) of incoming frames. When silence stretches over N sequential ticks (`SILENCE_THRESHOLD=150` for ~3 seconds of chunking), it wraps the accumulated stream into an in-memory or on-disk byte block, calling an asynchronous webhook (`on_audio_ready(filepath)`).

### 2. `ASREngine` (`llm_compression/sensors/asr_engine.py`)
- Core driver: `faster-whisper`.
- Operates primarily in locally optimized compute (via CTranslate2 on float16/int8 fallback). 
- Instantiated with standard `tiny/small` language configurations to optimize generation bounds.

### 3. `ActionManager` (`llm_compression/action/manager.py`)
- Provides an explicit high-level API (`execute("move", x=0, y=0)`) connecting to `pyautogui`.
- **Embodied Sync Hook:** On every executed action, the core intercepts the payload, constructs an English semantic explanation (`"Clicking here", "Typing..."`), and instantly pushes the generation request mapping to an injected `tts_engine`.
- Output audio invokes an inline `sounddevice.play()` call on a daemon thread, executing in synchronization alongside the synchronous GUI movement execution.

### 4. `SafetyMonitor` (`llm_compression/action/safety.py`)
- Provides resolution ceiling constraints to block unmapped cursor commands.
- Implements keyword blacklists to ignore destructive terminal or typing instructions.

## 3. Data Flows
1. User speaks: System records amplitude.
2. User stops speaking: VAD recognizes block completion.
3. System translates stream to `<Audio Path>`.
4. Transferred to `ASR`: `transcribe_sync`.
5. Received String -> parsed by matching conditions or future explicit Cognitive Engine strings.
6. Trigger `action.execute()`.
7. `action` invokes parallel thread TTS Generation -> Desktop Speaker playback.
8. `action` invokes `pyautogui` API -> Desktop Application manipulation.
9. End of cycle.
