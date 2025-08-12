"""
Utility functions for LogicIF framework, including LLM inference capabilities.
"""

import json
from typing import List, Dict, Any, Optional, Union


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file and return a list of dictionaries."""
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]


def write_jsonl(file_path: str, data: List[Dict[str, Any]]) -> None:
    """Write a list of dictionaries to a JSONL file."""
    with open(file_path, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')


def openai_inference(conversations: Union[List[List[Dict]], List[Dict]], 
                     model: str, 
                     return_json: bool = False, 
                     temperature: Optional[float] = None, 
                     reasoning_effort: str = 'medium', 
                     max_completion_tokens: Optional[int] = None) -> List[str]:
    """
    Enhanced OpenAI inference supporting reasoning models (o3, o3-mini, o1, o1-mini, o4-mini, etc.).
    - For reasoning models, supports reasoning_effort and max_completion_tokens.
    - For non-reasoning models, supports temperature as before.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Please install the OpenAI package: pip install openai")
    
    if not isinstance(conversations, list):
        conversations = [conversations]
    ret = []
    client = OpenAI()

    # List of reasoning model name substrings (case-insensitive)
    reasoning_model_keys = ["o3", "o1", "o4-mini"]
    def is_reasoning_model(model_name):
        model_name = model_name.lower()
        return any(key in model_name for key in reasoning_model_keys)

    for conv in conversations:
        try:
            kwargs = dict(
                model=model,
                messages=conv,
                response_format={"type": "json_object"} if return_json else None,
            )
            if is_reasoning_model(model):
                # Only add supported params for reasoning models
                if max_completion_tokens is not None:
                    kwargs["max_completion_tokens"] = max_completion_tokens
                if reasoning_effort is not None:
                    kwargs["reasoning_effort"] = reasoning_effort
                # Do NOT add temperature for reasoning models
            else:
                # For non-reasoning models, keep temperature if provided
                if temperature is not None:
                    kwargs["temperature"] = temperature
            # Remove None values (response_format)
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            completion = client.chat.completions.create(**kwargs)
            generation = completion.choices[0].message.content
        except Exception as e:
            print(f"Error during OpenAI inference: {str(e)}")
            generation = '[ERROR]'
        ret.append(generation)
    return ret


def anthropic_inference(conversations: Union[List[List[Dict]], List[Dict]], 
                       model: str, 
                       max_tokens: int = 8192, 
                       thinking_budget: Optional[int] = None, 
                       enable_thinking: bool = False) -> List[Union[str, Dict]]:
    """
    Anthropic Claude inference function.
    - Supports Claude models (claude-opus-4, claude-3.5-sonnet, claude-3-7-sonnet, etc.)
    - Uses the Anthropic API with proper message formatting
    - Supports extended thinking for compatible models (Claude 3.7+, Claude 4+)
    - Logs thinking content to console when enable_thinking=True
    - Returns: List of strings (normal mode) or List of dicts with 'text', 'thinking', 'has_thinking' keys (thinking mode)
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("Please install the anthropic package: pip install anthropic")
    
    if not isinstance(conversations, list):
        conversations = [conversations]
    
    ret = []
    client = anthropic.Anthropic()
    
    # Extended thinking support for compatible models
    def supports_thinking(model_name):
        thinking_models = ["claude-3-7", "claude-4", "claude-opus-4"]
        model_lower = model_name.lower()
        return any(thinking_model in model_lower for thinking_model in thinking_models)
    
    use_thinking = enable_thinking and supports_thinking(model)
    
    for conv in conversations:
        try:
            # Handle single conversation format
            if isinstance(conv, dict):
                conv = [conv]
            
            # Convert to Anthropic format if needed
            messages = []
            for msg in conv:
                if msg["role"] == "user":
                    messages.append({"role": "user", "content": msg["content"]})
                elif msg["role"] == "assistant":
                    messages.append({"role": "assistant", "content": msg["content"]})
            
            # Prepare API call
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages
            }
            
            # Add thinking parameters if supported
            if use_thinking:
                kwargs["extra_headers"] = {"anthropic-beta": "extended-thinking-2024-12-20"}
                if thinking_budget is not None:
                    kwargs["thinking_budget"] = thinking_budget
            
            response = client.messages.create(**kwargs)
            
            if use_thinking and hasattr(response, 'thinking'):
                # Extended thinking mode
                thinking_content = response.thinking if response.thinking else ""
                text_content = response.content[0].text if response.content else ""
                
                if enable_thinking and thinking_content:
                    print(f"[Thinking]: {thinking_content[:500]}...")
                
                result = {
                    "text": text_content,
                    "thinking": thinking_content,
                    "has_thinking": bool(thinking_content)
                }
            else:
                # Normal mode
                result = response.content[0].text if response.content else ""
            
            ret.append(result)
            
        except Exception as e:
            print(f"Error during Anthropic inference: {str(e)}")
            if use_thinking:
                ret.append({"text": "[ERROR]", "thinking": "", "has_thinking": False})
            else:
                ret.append("[ERROR]")
    
    return ret


def gemini_inference(conversations: Union[List[List[Dict]], List[Dict]], 
                    model: str, 
                    **kwargs) -> List[str]:
    """
    Google Gemini inference function.
    Note: This is a placeholder implementation. You'll need to implement
    the actual Gemini API integration based on your needs.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Please install the Google Generative AI package: pip install google-generativeai")
    
    if not isinstance(conversations, list):
        conversations = [conversations]
    
    ret = []
    
    for conv in conversations:
        try:
            # Convert conversation to Gemini format
            prompt = ""
            for msg in conv:
                if msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n"
            
            # This is a placeholder - implement actual Gemini API call
            generation = f"[Gemini response placeholder for: {prompt[:100]}...]"
            ret.append(generation)
            
        except Exception as e:
            print(f"Error during Gemini inference: {str(e)}")
            ret.append("[ERROR]")
    
    return ret 


def load_api_keys():
    """Load API keys from config.json and set as environment variables."""
    import os, json
    from pathlib import Path

    config_file = Path(__file__).parent / "config.json"
    if not config_file.exists():
        print("WARNING: config.json not found.")
        return False

    try:
        with open(config_file) as f:
            config = json.load(f)
        for k, v in config.items():
            if k.endswith('_API_KEY') and v:
                os.environ[k] = v
        return True
    except Exception as e:
        print(f"WARNING: Could not load config.json: {e}")
        return False