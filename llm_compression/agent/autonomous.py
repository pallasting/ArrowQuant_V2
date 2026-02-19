import time
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any

from llm_compression.sensors.manager import SensorManager
from llm_compression.action.manager import ActionManager
from llm_compression.inference.arrow_engine import ArrowEngine 

logger = logging.getLogger(__name__)

class AutonomousAgent:
    """
    The Integrated Self.
    
    Combines:
    - Sensors (Eyes/Ears) -> Log (Multimodal Input)
    - ArrowEngine (Brain) -> Logic/Decision/Planning & Memory/Embedding
    - ActionManager (Hands) -> Execution (PyAutoGUI)
    
    Operates on an OODA Loop:
    1. Observe: Gather multimodal data.
    2. Orient: Update context/memory.
    3. Decide: Generate action plan (ArrowEngine generative inference).
    4. Act: Execute tool calls.
    """
    
    def __init__(
        self, 
        workspace_dir: str, 
        engine: Optional[ArrowEngine] = None,
        enable_vision: bool = True,
        enable_action: bool = True
    ):
        self.workspace = Path(workspace_dir)
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        # 1. Sensors (Input)
        self.sensors = SensorManager(workspace_dir)
        if enable_vision:
            self.sensors.start_hardware()
            
        # 2. Hands (Output)
        self.actions = ActionManager(workspace_dir, sensor_manager=self.sensors)
        if not enable_action:
            # Maybe disable motor functions?
            pass
            
        # 3. Brain & Memory (ArrowEngine Zero-Copy Inference)
        self.engine = engine
        if not self.engine:
            logger.info("No ArrowEngine provided. Agent running in Headless/Mock mode.")

        self.short_term_memory: List[Dict] = []
        self.goal: Optional[str] = None
        self.running = False

    def stop(self):
        """Shutdown all subsystems."""
        self.running = False
        self.sensors.stop_hardware()
        # Clean up async resources if needed
        
    def observe(self) -> Dict[str, Any]:
        """
        Gather current state.
        Returns a snapshot of the world.
        """
        # 1. Vision
        visual_snapshot = None
        if self.sensors.vision:
             # Force a frame capture or get recent log
             recent = self.sensors.get_recent_events(limit=1)
             if recent and recent[0].get("modality") == "vision":
                 visual_snapshot = recent[0]
        
        # 2. Screen Context (from ActionManager)
        # ActionManager has access to screen size and PyAutoGUI
        screen_size = self.actions.screen_size
        
        return {
            "timestamp": time.time(),
            "visual": visual_snapshot,
            "screen_size": screen_size,
            "recent_actions": self.short_term_memory[-5:], # Last 5 actions
        }

    def orient(self, observation: Dict, goal: str) -> str:
        """
        Synthesize observation into a prompt context.
        """
        # Construct the "Prompt" for the LLM
        context = f"Goal: {goal}\n"
        context += f"Screen Size: {observation['screen_size']}\n"
        
        if observation['visual']:
            context += f"Visual Context: {observation['visual'].get('content', 'Unknown')}\n"
            
        context += "Recent History:\n"
        for item in observation['recent_actions']:
            context += f"- {item}\n"
            
        return context

    def decide(self, context: str) -> Dict:
        """
        Generate the next action using ArrowEngine.
        ArrowEngine utilizes zero-copy tiered loading of large model weights.
        """
        if not self.engine:
             logger.warning("No ArrowEngine! Returning dummy 'wait' action.")
             return {"action": "wait", "params": {}}

        prompt = (
            f"You are an AI Agent controlling a computer. \n"
            f"{context}\n"
            f"Available Actions: move(x,y), click(x,y), type(text), press(key), wait().\n"
            f"Output strictly strict JSON format: {{ \"action\": \"...\", \"params\": {{...}} }}\n"
            f"Response:"
        )

        try:
            # Expected future ArrowEngine generative capability
            if hasattr(self.engine, 'generate'):
                response_text = self.engine.generate(prompt, max_tokens=100)
            else:
                logger.warning("ArrowEngine.generate not implemented yet, returning dummy wait.")
                response_text = '{"action": "wait", "params": {}}'
                
            text = response_text.strip()
            # Basic cleanup if Markdown code blocks are used
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            return json.loads(text)
        except Exception as e:
            logger.error(f"Decision failed: {e}")
            return {"action": "wait", "params": {}}

    def act(self, decision: Dict) -> bool:
        """
        Execute the decided action.
        """
        action_type = decision.get("action")
        params = decision.get("params", {})
        
        if not action_type or action_type == "wait":
            return False
            
        success = self.actions.execute(action_type, **params)
        
        # Record to memory
        self.short_term_memory.append({
            "action": action_type,
            "params": params,
            "success": success,
            "timestamp": time.time()
        })
        
        return success

    def run_loop(self, goal: str, max_steps: int = 10):
        """
        Main Agent Loop.
        """
        self.goal = goal
        self.running = True
        step = 0
        
        logger.info(f"Agent starting goal: {goal}")
        
        # Ensure LLM connection
        # We need to manually handle async context for LLMClient if not used in 'async with'
        # But LLMClient.generate connects on demand via pool
        
        while self.running and step < max_steps:
            logger.info(f"--- Step {step} ---")
            
            # 1. Observe
            obs = self.observe()
            
            # 2. Orient
            context = self.orient(obs, goal)
            
            # 3. Decide
            decision = self.decide(context)
            logger.info(f"Decision: {decision}")
            
            # 4. Act
            self.act(decision)
            
            step += 1
            # Rate limit loop
            time.sleep(1.0) 
            
        logger.info("Agent finished.")
