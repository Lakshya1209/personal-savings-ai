"""
agents/base_agent.py — Groq LLM client manager, multi-agent LLM provider layer,
Pydantic validation, model failover, and graceful deterministic fallback execution.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional, Callable, Type
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load local environment variables from .env if present
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)

logger = logging.getLogger(__name__)

_CLIENT = None


def get_llm_client():
    """
    Returns an initialized Groq Client if GROQ_API_KEY is configured.
    Returns None if no key is present or invalid.
    """
    global _CLIENT
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.strip() == "" or api_key.strip() == "your_groq_api_key_here":
        return None
    
    if _CLIENT is None:
        try:
            from groq import Groq
            _CLIENT = Groq(api_key=api_key.strip())
        except Exception as e:
            logger.warning(f"Failed to initialize Groq Client: {e}")
            return None
    return _CLIENT


# Backward-compatible alias for existing imports
get_gemini_client = get_llm_client


def get_candidate_models() -> List[str]:
    """
    Returns an ordered list of verified candidate models:
    User-configured model first, followed by fast, reliable fallbacks.
    """
    configured = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()
    fallbacks = [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "groq/compound-mini",
        "qwen/qwen3.8-27b",
    ]
    candidates = [configured]
    for m in fallbacks:
        if m not in candidates:
            candidates.append(m)
    return candidates


def invoke_llm_structured(
    system_instruction: str,
    prompt: str,
    pydantic_schema: Type[BaseModel],
    tools: Optional[List[Callable]] = None,
    temperature: float = 0.1,
) -> Optional[BaseModel]:
    """
    Invokes Groq with JSON output mode and validates against a Pydantic schema.
    Embeds schema constraints in the system prompt.
    Features automatic model failover across verified Groq models.
    Returns validated Pydantic model or None if unavailable/failed.
    """
    client = get_llm_client()
    if client is None:
        return None

    # Extract schema structure to instruct the model cleanly
    try:
        schema_dict = pydantic_schema.model_json_schema()
        schema_str = json.dumps(schema_dict, indent=2)
    except Exception:
        schema_str = str(pydantic_schema.__annotations__)

    augmented_system = (
        f"{system_instruction.strip()}\n\n"
        f"CRITICAL REQUIREMENT: You MUST respond ONLY with a single valid JSON object that strictly adheres to this JSON schema:\n"
        f"{schema_str}\n"
        f"Do not include any explanation, introductory text, or markdown blocks outside the JSON."
    )

    candidate_models = get_candidate_models()

    for model_name in candidate_models:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": augmented_system},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
            )
            raw_text = response.choices[0].message.content
            if raw_text:
                cleaned_text = raw_text.strip()
                if cleaned_text.startswith("```"):
                    parts = cleaned_text.split("```")
                    if len(parts) >= 2:
                        cleaned_text = parts[1]
                    if cleaned_text.startswith("json"):
                        cleaned_text = cleaned_text[4:]
                    cleaned_text = cleaned_text.strip()
                
                data = json.loads(cleaned_text)
                # Unwrap nested 'json' wrapper if model wrapped it
                if isinstance(data, dict) and "json" in data and len(data) == 1 and isinstance(data["json"], dict):
                    data = data["json"]
                return pydantic_schema.model_validate(data)
        except ValidationError as val_err:
            logger.warning(f"Validation error on model {model_name}: {val_err}. Retrying once...")
            try:
                retry_messages = [
                    {"role": "system", "content": augmented_system},
                    {"role": "user", "content": prompt},
                    {
                        "role": "user",
                        "content": (
                            f"IMPORTANT: Your previous output did not match the required schema: {val_err}.\n"
                            f"Required JSON Schema:\n{schema_str}\n"
                            f"Return ONLY valid JSON matching this schema exactly."
                        ),
                    },
                ]
                response = client.chat.completions.create(
                    model=model_name,
                    messages=retry_messages,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                )
                raw_text = response.choices[0].message.content
                if raw_text:
                    cleaned = raw_text.strip()
                    if cleaned.startswith("```"):
                        parts = cleaned.split("```")
                        if len(parts) >= 2:
                            cleaned = parts[1]
                        if cleaned.startswith("json"):
                            cleaned = cleaned[4:]
                        cleaned = cleaned.strip()
                    data = json.loads(cleaned)
                    if isinstance(data, dict) and "json" in data and len(data) == 1 and isinstance(data["json"], dict):
                        data = data["json"]
                    return pydantic_schema.model_validate(data)
            except Exception as retry_err:
                logger.warning(f"Retry on {model_name} failed: {retry_err}")
                continue
        except Exception as e:
            logger.warning(f"Groq invocation error on {model_name}: {e}. Trying fallback if available...")
            continue

    return None


def invoke_llm_text(
    system_instruction: str,
    prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 1000,
) -> Optional[str]:
    """
    Invokes Groq for general conversational / educational text generation (used by FAQ Bot).
    Supports automatic model failover.
    """
    client = get_llm_client()
    if client is None:
        return None

    candidate_models = get_candidate_models()

    for model_name in candidate_models:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Groq text completion error on {model_name}: {e}. Trying fallback...")
            continue

    return None


# Backward-compatible function alias for all existing agents
invoke_gemini_structured = invoke_llm_structured
