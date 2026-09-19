"""
agents/base_agent.py — Gemini client manager, multi-agent LLM orchestration,
Pydantic validation, model failover, and graceful fallback execution.
"""

import os
import sys
import json
import logging

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Dict, Any, List, Optional, Callable, Type
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

# Load local environment variables from .env if present
load_dotenv(override=True)

logger = logging.getLogger(__name__)

_CLIENT = None


def get_gemini_client():
    """
    Returns an initialized google-genai Client if GEMINI_API_KEY is configured.
    Returns None if no key is present.
    """
    global _CLIENT
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key.strip() == "" or api_key.strip() == "your_gemini_api_key_here":
        return None
    
    if _CLIENT is None:
        try:
            from google import genai
            _CLIENT = genai.Client(api_key=api_key.strip())
        except Exception as e:
            logger.warning(f"Failed to initialize google-genai Client: {e}")
            return None
    return _CLIENT


def get_candidate_models() -> List[str]:
    """
    Returns an ordered list of verified candidate models:
    User configured model first, followed by verified fast and available flash models.
    """
    configured = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
    fallbacks = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.6-flash"]
    candidates = [configured]
    for m in fallbacks:
        if m not in candidates:
            candidates.append(m)
    return candidates


def invoke_gemini_structured(
    system_instruction: str,
    prompt: str,
    pydantic_schema: Type[BaseModel],
    tools: Optional[List[Callable]] = None,
    temperature: float = 0.1,
) -> Optional[BaseModel]:
    """
    Invokes Gemini with JSON output mode and validates against a Pydantic schema.
    Embeds schema constraints in prompt to remain 100% compliant with Gemini Developer API
    (avoiding 'additionalProperties' restrictions on Developer keys).
    Features automatic model failover across verified Gemini models.
    Returns validated Pydantic model or None if unavailable/failed.
    """
    client = get_gemini_client()
    if client is None:
        return None

    from google.genai import types

    # Extract schema structure to instruct the model cleanly
    try:
        schema_dict = pydantic_schema.model_json_schema()
        schema_str = json.dumps(schema_dict, indent=2)
    except Exception:
        schema_str = str(pydantic_schema.__annotations__)

    augmented_system = (
        f"{system_instruction.strip()}\n\n"
        f"CRITICAL: You MUST respond ONLY with a single valid JSON object that strictly adheres to this JSON schema:\n"
        f"{schema_str}\n"
        f"Do not include any introductory or concluding text outside the JSON."
    )

    config = types.GenerateContentConfig(
        system_instruction=augmented_system,
        temperature=temperature,
        response_mime_type="application/json",
    )

    candidate_models = get_candidate_models()

    for model_name in candidate_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            if response.text:
                cleaned_text = response.text.strip()
                if cleaned_text.startswith("```"):
                    cleaned_text = cleaned_text.split("```")[1]
                    if cleaned_text.startswith("json"):
                        cleaned_text = cleaned_text[4:]
                    cleaned_text = cleaned_text.strip()
                data = json.loads(cleaned_text)
                return pydantic_schema.model_validate(data)
        except ValidationError as val_err:
            logger.warning(f"Validation error on model {model_name}: {val_err}. Retrying once...")
            try:
                retry_prompt = (
                    f"{prompt}\n\n"
                    f"IMPORTANT: Your previous response failed schema validation: {val_err}.\n"
                    f"Required JSON Schema:\n{schema_str}\n"
                    f"Return ONLY valid JSON matching this schema."
                )
                response = client.models.generate_content(
                    model=model_name,
                    contents=retry_prompt,
                    config=config,
                )
                if response.text:
                    cleaned = response.text.strip()
                    if cleaned.startswith("```"):
                        cleaned = cleaned.split("```")[1]
                        if cleaned.startswith("json"):
                            cleaned = cleaned[4:]
                        cleaned = cleaned.strip()
                    data = json.loads(cleaned)
                    return pydantic_schema.model_validate(data)
            except Exception as retry_err:
                logger.warning(f"Retry on {model_name} failed: {retry_err}")
                continue
        except Exception as e:
            logger.warning(f"Gemini invocation error on {model_name}: {e}. Trying fallback if available...")
            continue

    return None
