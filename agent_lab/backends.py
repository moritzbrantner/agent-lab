"""Local Ollama transport; no inference policy lives in the runtime."""

import asyncio
import json
import urllib.error
import urllib.request
from dataclasses import replace
from urllib.parse import urlparse

from agent_lab.experiments import canonical
from agent_lab.runtime import (
    Reply,
    ResponseFailure,
    RuntimeFailure,
    TransientModelError,
)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "content": {"type": "string"},
        "calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["name", "arguments"],
            },
        },
    },
    "required": ["content", "calls"],
}


class OllamaAdapter:
    def __init__(self, endpoint="http://127.0.0.1:11434", *, response_schema=None):
        parsed = urlparse(endpoint)
        if parsed.scheme != "http" or parsed.hostname not in (
            "127.0.0.1",
            "localhost",
            "::1",
        ):
            raise ValueError("Ollama endpoint must be loopback HTTP")
        self.endpoint = endpoint.rstrip("/")
        self.response_schema = response_schema or RESPONSE_SCHEMA

    def optimize(self, config, requested):
        report = {}
        for name, value in requested.items():
            if name in ("context_size", "gpu_layers"):
                minimum = 1 if name == "context_size" else 0
                if type(value) is not int or not minimum <= value <= 131072:
                    raise ValueError("Invalid optimization bound")
                config = (
                    replace(config, context_size=value)
                    if name == "context_size"
                    else replace(config, options={**config.options, "num_gpu": value})
                )
                report[name] = {"status": "applied", "value": value}
            else:
                report[name] = {
                    "status": "unavailable",
                    "reason": "no measured control for pinned backend/model",
                }
        return config, report

    @staticmethod
    def _message(message):
        if message["role"] == "tool":
            return {
                "role": "user",
                "content": "Tool "
                + message["name"]
                + " returned: "
                + canonical(message["content"])
                + ". Continue with a JSON reply.",
            }
        if message.get("calls"):
            return {
                "role": message["role"],
                "content": canonical(
                    {"content": message["content"], "calls": message["calls"]}
                ),
            }
        return {"role": message["role"], "content": message["content"]}

    def _request(self, messages, config):
        options = {
            **config.options,
            "seed": config.seed,
            "temperature": config.temperature,
            "num_ctx": config.context_size,
            "num_predict": config.max_output_tokens,
        }
        body = {
            "model": config.model,
            "stream": False,
            "format": self.response_schema,
            "options": options,
            "messages": [self._message(message) for message in messages],
        }
        request = urllib.request.Request(
            self.endpoint + "/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=config.request_timeout) as response:
                value = json.load(response)
        except urllib.error.HTTPError as error:
            if error.code in (429, 503):
                raise TransientModelError(f"Backend HTTP {error.code}") from error
            raise RuntimeFailure(f"Backend HTTP {error.code}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeFailure(
                "Backend transport failed; completion unknown"
            ) from error
        usage = {
            "model_calls": 1,
            "input_tokens": value.get("prompt_eval_count"),
            "output_tokens": value.get("eval_count"),
            "measurements": {
                key: value[key] / 1e9
                for key in ("load_duration", "prompt_eval_duration", "eval_duration")
                if key in value
            },
        }
        try:
            parsed = json.loads(value["message"]["content"])
            if not isinstance(parsed, dict):
                raise RuntimeFailure("Backend reply must be an object")
            parsed.update(usage)
            return Reply.parse(parsed)
        except (KeyError, ValueError, TypeError, RuntimeFailure) as error:
            raise ResponseFailure("Malformed backend response", **usage) from error

    async def complete(self, messages, config):
        # Socket work is finite. Cancellation discards the result; no tool executes.
        return await asyncio.to_thread(self._request, messages, config)
