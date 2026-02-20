# Embodied Voice Agent - Requirements Document

## 1. Overview
The Embodied Voice Agent represents the next evolutionary step of AI-OS, transforming it from a text-based conversational interface into a fully autonomous, voice-controlled, embodied system capable of interacting with the physical operating system via dynamic UI manipulation. 

## 2. Core Objectives
1. **Continuous Voice Perception (ASR):** The OS should passively monitor microphone input safely, trigger voice-activity (VAD), and transcribe the user's speech accurately without active pushing of buttons.
2. **Embodied Action Control:** The OS should map intentions to direct OS UI interventions (moving the cursor, clicking, writing text).
3. **Synchronous Voice Narration (TTS):** To ensure transparency and human-like AI presence, anytime a physical action is planned or being executed, a multi-modal TTS component must narrate the action *concurrently* (fire-and-forget background audio thread).
4. **Safety & Sandboxing:** Ensure actions like moving the cursor or deleting directories are strictly bound by the OS resolution constraints and text pattern firewalls.
5. **Self-Evolution Ready:** The loop must save experience traces (`state`, `action`, `reward`) for future Reinforcement/Imitation Learning.

## 3. Key Functional Requirements (FRs)
- **FR1:** Implement an explicit `ASREngine` driven by `faster-whisper`.
- **FR2:** Support automatic language detection or explicit `zh` / `en` handling to ensure high accuracy of both English and Chinese dictation.
- **FR3:** Develop a secure `ActionManager` backed by `PyAutoGUI` for GUI actions.
- **FR4:** The `ActionManager` must accept a `TTSEngine` injector to synthesize and vocally narrate upcoming actions on asynchronous threads.
- **FR5:** Combine these systems in an End-to-End interactive shell (`demo_embodied_voice.py` or equivalent).

## 4. Non-Functional Requirements (NFRs)
- **NFR1 (Speed):** The ASR must use extremely fast local inference (`small` or `tiny` parameters) to minimize glass-to-glass latency.
- **NFR2 (Safety):** The Action mechanism requires a Fail-Safe interrupt (e.g., PyAutoGUI FAILSAFE) and boundary rectifications.
- **NFR3 (Modularity):** The architecture should be entirely decoupled so `CognitiveLoop` can eventually take over input processing.
