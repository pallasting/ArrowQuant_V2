import os
import json
import yaml
from pathlib import Path
from requests import post

# --- LLM API Configuration (from config.yaml and environment) ---
def load_config():
    """Load configuration from config.yaml and environment variables."""
    config_path = Path(__file__).parent.parent / "config.yaml"

    # Load from config file
    config = {}
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}

    # Override with environment variables if present
    llm_api_key = os.getenv("LLM_API_KEY") or os.getenv("LLM_CLOUD_API_KEY")
    llm_base_url = os.getenv("LLM_BASE_URL")

    # Use environment variables or fall back to config
    # Priority: ENV > config.yaml > default (localhost:8045 - Antigravity Manager)
    api_key = llm_api_key or config.get('llm', {}).get('cloud_api_key')

    # Default to Antigravity Manager API (verified working)
    default_base_url = 'http://localhost:8045'
    base_url = llm_base_url or config.get('llm', {}).get('cloud_endpoint', default_base_url)

    return {
        'api_key': api_key,
        'base_url': base_url,
        'endpoint': '/v1/chat/completions',  # ✅ OpenAI protocol (verified working)
        'model': 'gpt-4',  # Antigravity Manager 支持的模型
        'protocol': 'openai'  # OpenAI protocol format
    }

CONFIG = load_config()
LLM_API_KEY = CONFIG['api_key']
LLM_BASE_URL = CONFIG['base_url']
LLM_ENDPOINT = CONFIG['endpoint']
LLM_MODEL = CONFIG['model']

def llm_call_for_compression(prompt: str) -> str:
    """
    Calls the configured LLM API to get a response.

    Uses OpenAI protocol format (verified working with Gemini CLI OAuth).
    """
    if not LLM_API_KEY:
        return "Error: No API key configured. Set LLM_API_KEY environment variable or configure in config.yaml"

    full_url = f"{LLM_BASE_URL}{LLM_ENDPOINT}"

    # OpenAI protocol format
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}"  # ✅ OpenAI auth format
    }
    payload = {
        "model": LLM_MODEL,
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = post(full_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        json_response = response.json()

        # OpenAI protocol response format
        if "choices" in json_response and len(json_response["choices"]) > 0:
            content = json_response["choices"][0]["message"]["content"]
            return content
        elif "error" in json_response:
            error_msg = json_response["error"].get("message", str(json_response["error"]))
            print(f"API Error: {error_msg}")
            return f"Error: {error_msg}"
        else:
            # Fallback for unexpected response format
            return str(json_response)

    except Exception as e:
        print(f"Error calling LLM for compression: {e}")
        return f"Error: {e}"

def compress_text_generative(text: str) -> dict:
    """
    Compresses text using an LLM to generate a summary and extract key entities.
    """
    print(f"Calling LLM for text summary...")
    prompt_summary = f"Summarize the following text concisely. Text: {text}"
    summary = llm_call_for_compression(prompt_summary)

    print(f"Calling LLM for entity extraction...")
    # Request entities in JSON list format for easier parsing
    prompt_entities = f"Extract key entities (e.g., all numbers, names, dates, locations, key terms) from the following text and return as a JSON list of strings. Example: ['entity1', 'entity2']. Text: {text}"
    entities_str = llm_call_for_compression(prompt_entities)

    entities_list = []
    try:
        # Try to parse as JSON list
        parsed_entities = json.loads(entities_str)
        if isinstance(parsed_entities, list):
            entities_list = parsed_entities
        else:
            print(f"Warning: LLM returned non-list JSON for entities: {entities_str[:200]}")
            entities_list = [entities_str[:200]]
    except json.JSONDecodeError:
        print(f"Warning: LLM did not return valid JSON for entities: {entities_str[:200]}")
        entities_list = [entities_str[:200]]

    # Calculate compression metrics
    original_len = len(text.encode('utf-8'))
    summary_len = len(summary.encode('utf-8'))

    return {
        "summary": summary,
        "entities": entities_list,
        "original_length": original_len,
        "compressed_length_estimate": summary_len,
        "llm_model": LLM_MODEL,
        "status": "LLM Compressed"
    }

def mock_compress_text_generative(text: str) -> dict:
    """A simple, non-LLM based compression for comparison."""
    summary = text[:min(len(text), 50)] + "..."
    entities = ["mock_entity_1", "mock_entity_2"]
    original_len = len(text.encode('utf-8'))
    compressed_len = len(summary.encode('utf-8'))
    return {
        "summary": summary,
        "entities": entities,
        "original_length": original_len,
        "compressed_length_estimate": compressed_len,
        "llm_model": "Mock",
        "status": "Mock Compressed"
    }
