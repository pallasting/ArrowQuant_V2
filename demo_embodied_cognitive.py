import asyncio
import time
import os
import sounddevice as sd
import numpy as np
import re
import json

from llm_compression.sensors.audio import AudioModule
from llm_compression.sensors.asr_engine import ASREngine
from llm_compression.expression.tts.tts_engine import TTSEngine
from llm_compression.expression.expression_types import TTSConfig, TTSBackend
from llm_compression.action.manager import ActionManager
from llm_compression.llm_client import LLMClient, ArrowLLMClient
from llm_compression.compressor import LLMCompressor
from llm_compression.conversational_agent import ConversationalAgent
from llm_compression.inference.arrow_engine import ArrowEngine

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


EMBODIED_SYSTEM_PROMPT = """You are AI-OS, an intelligent embodied agent operating on the user's computer. 
You can understand the user's voice and interact with the graphical operating system.

If the user asks you to perform a physical action on the screen (like moving the cursor, clicking, or typing text), you must output a JSON action block at the VERY END of your response.
Your response MUST be in this exact format:
[Your conversational response that you want spoken aloud to the user in English]
[ACTION: {"type": "move", "params": {"x": 500, "y": 500}}]

Supported Actions:
1. move cursor to center: [ACTION: {"type": "move", "params": {"x": "center", "y": "center"}}] 
2. click: [ACTION: {"type": "click", "params": {"clicks": 1}}]
3. type text: [ACTION: {"type": "type", "params": {"text": "hello"}}]

Only output ONE ACTION block at the end if an action is required. If no Action is needed, just reply conversationally without the ACTION block.
Keep your conversational response extremely short, concise, and casual (maximum 1-2 sentences)."""

def tts_play(tts_engine, text):
    """Play TTS audibly."""
    try:
        chunks = list(tts_engine.synthesize(text))
        if chunks:
            arr = np.concatenate(chunks)
            sd.play(arr, samplerate=tts_engine.config.sample_rate)
    except Exception as e:
        print(f"Failed to play TTS: {e}")

async def run_embodied_agent():
    print("🧠 Loading AI-OS Cognitive & Cortex Modules...")

    # Load Action & TTS layer
    config = TTSConfig(backend=TTSBackend.PIPER, streaming=False, cache_enabled=False)
    tts = TTSEngine(config)
    
    os.makedirs("workspace", exist_ok=True)
    action_manager = ActionManager("workspace", tts_engine=tts)

    # Load LLM Core (Arrow Native Engine Memory vs Remote LLM API)
    print("🔗 Connecting to native ArrowEngine Inference Kernel...")
    model_path = "llm_compression/models/llm_native_model"
    
    # Check if a model exists natively in Arrow format
    if os.path.exists(model_path):
        print("  -> Booting localized memory state via Apache Arrow (Zero-Copy) engine.")
        arrow_engine = ArrowEngine(model_path=model_path, device="auto")
        llm_client = ArrowLLMClient(arrow_engine)
    else:
        print(f"  -> ⚠️ Target {model_path} missing Arrow model chunks.")
        print("  -> Initiating Fallback (Ollama API Layer Emulator) for local development testing...")
        llm_client = LLMClient(endpoint="http://localhost:11434", api_type="ollama")
        
    compressor = LLMCompressor(llm_client)
    agent = ConversationalAgent(
        llm_client=llm_client,
        compressor=compressor,
        user_id="embodied_user",
        enable_personalization=False
    )

    # Load Auditory Layer
    asr = ASREngine(model_size="small", device="auto")
    
    # Event loop queues bridging callback to async flow
    audio_queue = asyncio.Queue()

    def on_voice_command(filepath):
        # ASR callback pushes file to async queue
        asyncio.run_coroutine_threadsafe(audio_queue.put(filepath), asyncio.get_running_loop())

    print("\n🎤 Starting Auditory Microphone Capture...")
    mic = AudioModule(output_dir="workspace", on_audio_ready=on_voice_command)
    mic.start()
    
    tts_play(tts, "Cognitive capabilities online. I am ready to assist.")
    print("\n=============================================")
    print("🎙️ AI-OS Cognitive Agent Live!")
    print("Try asking it: 'Please move the cursor to the center of the screen.'")
    print("Or: 'Type hello world for me.'")
    print("Press Ctrl+C to stop.")
    print("=============================================\n")
    
    try:
        while True:
            # Wait for user speech file
            filepath = await audio_queue.get()
            
            print(f"\n[ASR] Transcribing Voice Command...")
            try:
                text = asr.transcribe_sync(filepath, language="zh") # Allows mix
                print(f"👂 [User Said]: {text}")
                
                # Let Cognitive Agent Think
                print("🤔 [Cognitive Loop] Processing intent...")
                # Note: Time the chat execution
                response = await agent.chat(
                    user_message=text, 
                    max_context_turns=3,
                    system_prompt=EMBODIED_SYSTEM_PROMPT
                )
                
                output = response.message.strip()
                print(f"🗣️ [Agent Reply]:\n{output}\n------------------------")
                
                # Parse ACTION block
                action_block_match = re.search(r'\[ACTION:\s*(\{.*?\})\s*\]', output, re.DOTALL)
                
                clean_speech = output
                if action_block_match:
                    action_json_str = action_block_match.group(1)
                    clean_speech = output[:action_block_match.start()].strip()
                    
                    try:
                        action_data = json.loads(action_json_str)
                        action_type = action_data.get("type", "")
                        params = action_data.get("params", {})
                        
                        # Translate 'center' to screen coordinates dynamically
                        if "x" in params and params["x"] == "center":
                            import pyautogui
                            w, h = pyautogui.size()
                            params["x"] = w // 2
                            params["y"] = h // 2
                            
                        print(f"🚀 [Executing Action] >> {action_type} | params: {params}")
                        
                        # We do NOT run `tts_play` here blocking the script. 
                        # We let the `ActionManager` narrate it inherently, or just narrate the clean speech first!
                        # Let's narrate the clean speech!
                        if clean_speech:
                            tts_play(tts, clean_speech)
                            
                        action_manager.execute(action_type, **params)
                        
                    except json.JSONDecodeError:
                        print(f"⚠️ Failed to parse Action JSON: {action_json_str}")
                else:
                    # No explicit action, just talk
                    if clean_speech:
                        tts_play(tts, clean_speech)
                        
            except Exception as e:
                print(f"Error handling cognitive loop: {e}")
                
    except asyncio.CancelledError:
        pass
    finally:
        mic.stop()
        print("\nShutting down AI-OS Cognitive Agent.")

if __name__ == "__main__":
    try:
        asyncio.run(run_embodied_agent())
    except KeyboardInterrupt:
        pass
