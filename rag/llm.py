"""Local LLM access through Ollama's HTTP API."""
import base64
import os
import time

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
TEXT_MODEL = os.environ.get("RAG_TEXT_MODEL", "llama3.1:8b")
VISION_MODEL = os.environ.get("RAG_VISION_MODEL", "llava:7b")


class OllamaClient:
    def __init__(self, model=TEXT_MODEL, url=OLLAMA_URL, num_ctx=4096, timeout=300):
        self.model, self.url, self.num_ctx, self.timeout = model, url.rstrip("/"), num_ctx, timeout

    def chat(self, messages, images=None, max_tokens=300):
        """Return (text, stats). temperature 0 and a fixed seed keep runs repeatable."""
        msgs = [dict(m) for m in messages]
        if images:
            msgs[-1]["images"] = [base64.b64encode(i).decode() for i in images]
        body = {"model": self.model, "messages": msgs, "stream": False,
                "options": {"temperature": 0, "seed": 0, "num_ctx": self.num_ctx, "num_predict": max_tokens}}
        t0 = time.perf_counter()
        r = requests.post(f"{self.url}/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        stats = {"wall_ms": (time.perf_counter() - t0) * 1000,
                 "prompt_tokens": data.get("prompt_eval_count"), "output_tokens": data.get("eval_count")}
        return data["message"]["content"].strip(), stats

    def available(self):
        try:
            tags = requests.get(f"{self.url}/api/tags", timeout=3).json()
            return any(m["name"].startswith(self.model.split(":")[0]) for m in tags.get("models", []))
        except requests.RequestException:
            return False


class FakeLLM:
    """Test double: returns canned text and records the messages it was given."""

    def __init__(self, reply="Fake answer [FIST 3-16, p.1]."):
        self.reply, self.calls = reply, []

    def chat(self, messages, images=None, max_tokens=300):
        self.calls.append({"messages": messages, "images": images})
        return self.reply, {"wall_ms": 0.0, "prompt_tokens": 0, "output_tokens": 0}
