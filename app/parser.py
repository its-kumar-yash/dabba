"""Order parser backends: TinkerBackend (hosted demo) and OllamaBackend (laptop)."""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.schema import FIELDS, build_system_prompt, validate  # noqa: E402

JSON_RE = re.compile(r"\{.*\}", re.S)


class ParseError(Exception):
    pass


def extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = JSON_RE.search(text)
    if not m:
        raise ParseError(f"no JSON in reply: {text[:200]!r}")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise ParseError(f"bad JSON: {e}: {m.group(0)[:200]!r}")
    if not isinstance(obj, dict):
        raise ParseError("JSON is not an object")
    obj = {k: obj.get(k) for k in FIELDS}
    obj["items"] = obj.get("items") or []
    obj["prefs"] = obj.get("prefs") or []
    for it in obj["items"]:
        if isinstance(it, dict) and isinstance(it.get("qty"), (str, float)):
            try:
                it["qty"] = int(float(it["qty"]))
            except ValueError:
                pass
    errs = validate(obj)
    if errs:
        raise ParseError(f"invalid fields {errs}: {obj}")
    return obj


class ParserBackend:
    name = "base"

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError

    def parse(self, text: str, hints: list[str] | None = None) -> tuple[dict, int]:
        t0 = time.perf_counter()
        label = extract_json(self.complete(build_system_prompt(hints), text))
        return label, int((time.perf_counter() - t0) * 1000)


class OllamaBackend(ParserBackend):
    name = "ollama"

    def __init__(self):
        self.model = os.environ.get("OLLAMA_MODEL", "dabba")
        self.host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")

    def complete(self, system: str, user: str) -> str:
        r = requests.post(f"{self.host}/api/chat", json={"model": self.model, "stream": False, "think": False,
                          "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                          "options": {"temperature": 0, "num_predict": 300}}, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"]


class TinkerBackend(ParserBackend):
    name = "tinker"

    def __init__(self, sampler_path: str | None = None):
        from app.tinker_client import TinkerSampler

        self.sampler = TinkerSampler(model_path=sampler_path or os.environ["DABBA_SAMPLER_PATH"], max_tokens=300, temperature=0.0)

    def complete(self, system: str, user: str) -> str:
        return self.sampler.complete(system, user)


def get_backend() -> ParserBackend:
    return TinkerBackend() if os.environ.get("PARSER_BACKEND", "tinker").lower() == "tinker" else OllamaBackend()
