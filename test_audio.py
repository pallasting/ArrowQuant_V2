import asyncio
from llm_compression.expression_layer import MultiModalExpressor
from llm_compression.llm_client import LLMClient
from llm_compression.reconstructor import LLMReconstructor
from llm_compression.memory_primitive import MemoryPrimitive
import numpy as np

class MockLLMClient:
    async def generate(self, prompt, **kwargs):
        class Result:
            text = "Welcome to the AI Operating System! Voice synthesis is online."
        return Result()

class MockReconstructor:
    async def reconstruct(self, content):
        class Result:
            full_text = content
        return Result()

async def main():
    expressor = MultiModalExpressor(MockLLMClient(), MockReconstructor())
    
    # Needs some memory to pass type checks
    mem = MemoryPrimitive(id="m1", content="Hello", embedding=np.zeros(2))
    
    print("Generating audio...")
    result = await expressor.express_audio(memories=[mem])
    print(f"Result Modality: {result.modality}")
    print(f"Result Content Path: {result.content}")

if __name__ == "__main__":
    asyncio.run(main())
