# ai_models.py
"""
Contains classes for interacting with different AI model APIs.
FIXED: Restored LOCAL_MODEL_NAME constant.
FIXED: Conditional 'response_format' for LocalModel (only sends if requested).
LAZY LOADING ENABLED: Vertex credentials are only checked upon generation.
ADDED: GeminiAPIModel for standard API Key authentication.
"""
import os
import json
import logging
import requests
from abc import ABC, abstractmethod
import datetime
from google import genai
from google.genai import types

# --- Constants and Setup ---
VERTEXAI_PROJECT = "qtow-148623"
VERTEXAI_LOCATION = "global"
VERTEXAI_CRED_FILE = 'cred.json'
LOCAL_MODEL_ENDPOINT = "http://localhost:1234/v1/chat/completions"
LOCAL_MODEL_NAME = "qwen2-0.5b-instruct" # Restored constant
DEFAULT_SYSTEM_INSTRUCTION = "You are a helpful AI Assistant."
logger = logging.getLogger(__name__)

# --- Model Definitions ---
MODEL_TOKEN_LIMITS = {
    "gemini-2.5-pro": 2097152,
    "gemini-2.5-flash": 1048576,
    "Local": 32768
}

MODEL_CAPABILITIES = {
    "gemini-2.5-flash": {"supports_thinking": True},
    "gemini-2.5-pro":   {"supports_thinking": True},
    "Local":                {"supports_thinking": False}
}

class AIModel(ABC):
    def __init__(self, model_name, system_instruction, debug=False):
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.debug = debug
        self.token_limit = MODEL_TOKEN_LIMITS.get(model_name, 1048576)

    @abstractmethod
    def generate(self, user_prompt, **kwargs) -> dict: pass

    @abstractmethod
    def count_tokens(self, content: str) -> int: pass


class VertexAIModel(AIModel):
    """Handles interaction with Google's Vertex AI using the modern genai SDK."""
    _client = None

    def __init__(self, model_name, system_instruction, debug=False):
        super().__init__(model_name, system_instruction, debug)
        logger.info(f"VertexAIModel context established for model: {self.model_name} (Lazy Load)")

    @classmethod
    def _ensure_initialized(cls):
        if cls._client is None:
            if not os.path.exists(VERTEXAI_CRED_FILE):
                raise FileNotFoundError(f"Vertex AI credentials not found at '{VERTEXAI_CRED_FILE}'.")
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = VERTEXAI_CRED_FILE
            cls._client = genai.Client(vertexai=True, project=VERTEXAI_PROJECT, location=VERTEXAI_LOCATION)
            logger.info("google.genai.Client initialized successfully for Vertex AI.")

    def count_tokens(self, content: str) -> int:
        if not content: return 0
        try:
            self._ensure_initialized()
            response = self._client.models.count_tokens(model=self.model_name, contents=[content])
            return response.total_tokens
        except Exception as e:
            logger.warning(f"Token counting via API failed: {e}. Falling back to estimate.")
            return len(content) // 4

    def _serialize_chunk(self, chunk) -> dict:
        serialized = {"text": getattr(chunk, 'text', '')}
        if hasattr(chunk, 'usage_metadata'):
            usage = chunk.usage_metadata
            serialized['usage_metadata'] = {
                'prompt_token_count': usage.prompt_token_count,
                'candidates_token_count': usage.candidates_token_count,
                'total_token_count': usage.total_token_count,
                'thoughts_token_count': getattr(usage, 'thoughts_token_count', 0)
            }
        return serialized

    def generate(self, user_prompt, **kwargs) -> dict:
        try:
            self._ensure_initialized()
        except Exception as e:
             return {"text": f"Error: Vertex AI initialization failed - {e}", "input_tokens": 0, "output_tokens": 0, "full_response_log": []}

        if self.debug:
            try:
                debug_folder = "raw_debug_folder"
                os.makedirs(debug_folder, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                safe_model_name = self.model_name.replace('/', '_').replace(':', '_')
                filename = os.path.join(debug_folder, f"{timestamp}_{safe_model_name}_prompt.txt")
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(user_prompt)
                    if "response_schema" in kwargs:
                        f.write(f"\n\n--- SCHEMA ---\n{kwargs['response_schema']}")
            except Exception: pass

        safety_settings = [types.SafetySetting(category=c, threshold="BLOCK_NONE") for c in ["HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_DANGEROUS_CONTENT", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_HARASSMENT"]]
        thinking_config=types.ThinkingConfig(
      thinking_budget=8192,
    ),
        system_instruction_part = types.Part.from_text(text=self.system_instruction) if self.system_instruction else None

        config_params = {
            "temperature": kwargs.get("temperature", 0.5),
            "top_p": kwargs.get("top_p", 0.75),
            "max_output_tokens": kwargs.get("max_output_tokens", 63000),
            "safety_settings": safety_settings,
            "thinking_config": thinking_config,
            "system_instruction": system_instruction_part,
        }

        if "response_schema" in kwargs:
            config_params["response_schema"] = kwargs["response_schema"]
            config_params["response_mime_type"] = kwargs.get("response_mime_type", "application/json")
        elif "response_mime_type" in kwargs:
             config_params["response_mime_type"] = kwargs["response_mime_type"]

        model_caps = MODEL_CAPABILITIES.get(self.model_name, {})
        if model_caps.get("supports_thinking", False):
            config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=-1)

        generation_config = types.GenerateContentConfig(**config_params)
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])]

        full_text_parts, all_chunks_serialized, final_chunk = [], [], None
        try:
            response_stream = self._client.models.generate_content_stream(model=self.model_name, contents=contents, config=generation_config)
            for chunk in response_stream:
                full_text_parts.append(chunk.text)
                all_chunks_serialized.append(self._serialize_chunk(chunk))
                final_chunk = chunk
            
            if not final_chunk: return {"text": "Error: Empty response stream"}

            full_text_str = "".join(part for part in full_text_parts if part is not None)
            metadata = final_chunk.usage_metadata
            return {"text": full_text_str, "input_tokens": metadata.prompt_token_count, "output_tokens": metadata.candidates_token_count, "thoughts_tokens": getattr(metadata, 'thoughts_token_count', 0), "full_response_log": all_chunks_serialized}
        except Exception as e:
            logger.error(f"Vertex AI API call failed: {e}", exc_info=self.debug)
            return {"text": f"Error: Vertex AI call failed - {e}", "input_tokens": 0, "output_tokens": 0}


class GeminiAPIModel(AIModel):
    """Handles interaction with Google's Gemini API using an API Key (Non-Vertex)."""
    _client = None

    def __init__(self, model_name, system_instruction, debug=False):
        super().__init__(model_name, system_instruction, debug)
        self._ensure_initialized()
        logger.info(f"GeminiAPIModel context established for model: {self.model_name}")

    @classmethod
    def _ensure_initialized(cls):
        if cls._client is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                logger.warning("GEMINI_API_KEY not found in environment. GeminiAPIModel may fail.")
                # We don't raise here to allow instantiation, but generation will likely fail or require key injection later.
                return
            
            # Initialize client with api_key. 
            # Note: explicit 'vertexai=False' is implied when using api_key with the unified SDK.
            cls._client = genai.Client(api_key=api_key)
            logger.info("google.genai.Client initialized successfully for Gemini API (Key-based).")

    def count_tokens(self, content: str) -> int:
        if not content: return 0
        try:
            if self._client is None: self._ensure_initialized()
            response = self._client.models.count_tokens(model=self.model_name, contents=[content])
            return response.total_tokens
        except Exception as e:
            logger.warning(f"Token counting via API failed: {e}. Falling back to estimate.")
            return len(content) // 4

    def _serialize_chunk(self, chunk) -> dict:
        serialized = {"text": getattr(chunk, 'text', '')}
        if hasattr(chunk, 'usage_metadata'):
            usage = chunk.usage_metadata
            serialized['usage_metadata'] = {
                'prompt_token_count': usage.prompt_token_count,
                'candidates_token_count': usage.candidates_token_count,
                'total_token_count': usage.total_token_count,
                'thoughts_token_count': getattr(usage, 'thoughts_token_count', 0)
            }
        return serialized

    def generate(self, user_prompt, **kwargs) -> dict:
        try:
            if self._client is None: self._ensure_initialized()
            if self._client is None: raise ValueError("Client not initialized. Check GEMINI_API_KEY.")
        except Exception as e:
             return {"text": f"Error: Gemini API initialization failed - {e}", "input_tokens": 0, "output_tokens": 0, "full_response_log": []}

        if self.debug:
            try:
                debug_folder = "raw_debug_folder"
                os.makedirs(debug_folder, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                safe_model_name = self.model_name.replace('/', '_').replace(':', '_')
                filename = os.path.join(debug_folder, f"{timestamp}_{safe_model_name}_prompt.txt")
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(user_prompt)
                    if "response_schema" in kwargs:
                        f.write(f"\n\n--- SCHEMA ---\n{kwargs['response_schema']}")
            except Exception: pass

        safety_settings = [types.SafetySetting(category=c, threshold="BLOCK_NONE") for c in ["HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_DANGEROUS_CONTENT", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_HARASSMENT"]]
        system_instruction_part = types.Part.from_text(text=self.system_instruction) if self.system_instruction else None

        config_params = {
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.95),
            "max_output_tokens": kwargs.get("max_output_tokens", 64000),
            "safety_settings": safety_settings,
            "system_instruction": system_instruction_part,
        }

        if "response_schema" in kwargs:
            config_params["response_schema"] = kwargs["response_schema"]
            config_params["response_mime_type"] = kwargs.get("response_mime_type", "application/json")
        elif "response_mime_type" in kwargs:
             config_params["response_mime_type"] = kwargs["response_mime_type"]

        model_caps = MODEL_CAPABILITIES.get(self.model_name, {})
        if model_caps.get("supports_thinking", False):
            config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=-1)

        generation_config = types.GenerateContentConfig(**config_params)
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])]

        full_text_parts, all_chunks_serialized, final_chunk = [], [], None
        try:
            response_stream = self._client.models.generate_content_stream(model=self.model_name, contents=contents, config=generation_config)
            for chunk in response_stream:
                full_text_parts.append(chunk.text)
                all_chunks_serialized.append(self._serialize_chunk(chunk))
                final_chunk = chunk
            
            if not final_chunk: return {"text": "Error: Empty response stream"}

            full_text_str = "".join(part for part in full_text_parts if part is not None)
            metadata = final_chunk.usage_metadata
            return {"text": full_text_str, "input_tokens": metadata.prompt_token_count, "output_tokens": metadata.candidates_token_count, "thoughts_tokens": getattr(metadata, 'thoughts_token_count', 0), "full_response_log": all_chunks_serialized}
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}", exc_info=self.debug)
            return {"text": f"Error: Gemini API call failed - {e}", "input_tokens": 0, "output_tokens": 0}


class LocalModel(AIModel):
    def __init__(self, model_name_override, system_instruction, debug=False, api_endpoint_override=None):
        super().__init__(model_name_override, system_instruction, debug)
        self.api_url = api_endpoint_override or LOCAL_MODEL_ENDPOINT
        logger.info(f"LocalModel initialized for model '{self.model_name}' at endpoint: {self.api_url}")

    def count_tokens(self, content: str) -> int:
        return len(content) // 4 if content else 0

    def generate(self, user_prompt, **kwargs) -> dict:
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "system", "content": self.system_instruction}, {"role": "user", "content": user_prompt}],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_output_tokens", 32000)
        }
        
        # --- CONDITIONAL FORMATTING ---
        # Only add response_format if specifically requested (e.g., for JSON mode).
        # This prevents 400 Errors on simple summary requests.
        if "response_format" in kwargs:
             payload["response_format"] = kwargs["response_format"]
        
        if self.debug:
            logger.debug(f"--- Local AI Call ---\nURL: {self.api_url}\nPayload: {json.dumps(payload, indent=2)}\n-----------------")
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage", {})
            return {"text": data['choices'][0]['message']['content'], "input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0), "thoughts_tokens": 0, "full_response_log": [data]}
        except requests.RequestException as e:
            logger.error(f"Local AI API call failed: {e}", exc_info=self.debug)
            return {"text": f"Error: Local AI call failed - {e}", "input_tokens": 0, "output_tokens": 0}