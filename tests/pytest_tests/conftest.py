import os
from dotenv import load_dotenv, find_dotenv
from deepeval.evaluate import AsyncConfig
from deepeval.models import OllamaModel
import os
from typing import Optional, Tuple, Union
from pydantic import BaseModel
import uuid
import requests
import json
import pytest
import deepeval

load_dotenv(find_dotenv())

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8080")

class OllamaModelNoThink(OllamaModel):
     def generate(self, prompt: str, schema: Optional[BaseModel] = None) -> Tuple[Union[str, BaseModel], float]:
        chat_model = self.load_model()
        messages = [{"role": "user", "content": prompt}]

        response = chat_model.chat(
            model=self.name,
            messages=messages,
            format=schema.model_json_schema() if schema else None,
            options={
                **{"temperature": self.temperature},
                **self.generation_kwargs,
            },
            think=False
        )
        return (
            (
                schema.model_validate_json(response.message.content)
                if schema
                else response.message.content
            ),
            0,
        )

     async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None) -> Tuple[Union[str, BaseModel], float]:
        chat_model = self.load_model(async_mode=True)
        messages = [{"role": "user", "content": prompt}]

        response = await chat_model.chat(
            model=self.name,
            messages=messages,
            format=schema.model_json_schema() if schema else None,
            options={
                **{"temperature": self.temperature},
                **self.generation_kwargs,
            },
            think=False
        )
        return (
            (
                schema.model_validate_json(response.message.content)
                if schema
                else response.message.content
            ),
            0,
        )

## -- Application Specific methods ---

def chat(message: str, session_id: str) -> str:
    """
    Call the streaming chat endpoint and collect the full assistant response.
    Returns the concatenated text content.
    """
    resp = requests.post(
        f"{BACKEND_URL}/api/chat/stream",
        json={"message": message, "session_id": session_id},
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()

    full_text = ""
    for line in resp.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8") if isinstance(line, bytes) else line
        if not decoded.startswith("data: "):
            continue
        payload = decoded[6:].strip()
        if not payload:
            continue
        try:
            data = json.loads(payload)
            if data.get("type") == "chunk":
                full_text += data.get("content", "")
        except json.JSONDecodeError:
            pass
    return full_text.strip()


def get_cart(session_id: str) -> dict:
    resp = requests.get(f"{BACKEND_URL}/api/cart/{session_id}", timeout=10)
    return resp.json()

def clear_cart(session_id:str):
    requests.delete(f"{BACKEND_URL}/api/cart/{session_id}/clear", timeout=10)

def rag_search(query: str) -> list[dict]:
    resp = requests.get(
        f"{BACKEND_URL}/api/products/search/semantic", 
        params={"q": query }, 
        timeout=30
    )
    return resp.json()

### ----- Shared Fixtures of PyTest used across the tests ---------

# Hook of Pytest conftest file
def pytest_configure():
    api_key = os.getenv("CONFIDENT_AI_API_PORTAL")
    if api_key:
        deepeval.login(api_key=api_key)
        print("Confident AI reporting enabled; DeepEval metrics will be submitted.")
    else:
        print("Confident AI reporting disabled; set CONFIDENT_AI_API_PORTAL to submit metrics.")


@pytest.fixture(scope="session")
def session_id() -> str:
    return f"test_{uuid.uuid4().hex[:12]}"


@pytest.fixture(scope="session")
def judge():
    return OllamaModelNoThink(
        model=os.getenv("LOCAL_OLLAMA_MODEL"), 
        base_url=os.getenv("LOCAL_OLLAMA_BASE_URL")
    )
