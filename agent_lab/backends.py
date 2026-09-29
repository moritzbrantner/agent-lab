"""Local Ollama transport; no inference policy lives in the runtime."""

import asyncio
import json
import urllib.error
import urllib.request
from urllib.parse import urlparse

from agent_lab.runtime import Reply, RuntimeFailure, TransientModelError

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
    def __init__(self, endpoint="http://127.0.0.1:11434"):
        parsed = urlparse(endpoint)
        if parsed.scheme != "http" or parsed.hostname not in (
            "127.0.0.1",
            "localhost",
            "::1",
        ):
            raise ValueError("Ollama endpoint must be loopback HTTP")
        self.endpoint = endpoint.rstrip("/")

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
            "format": RESPONSE_SCHEMA,
            "options": options,
            "messages": [
                {
                    "role": message["role"],
                    "content": (
                        message["content"]
                        if isinstance(message["content"], str)
                        else json.dumps(message["content"])
                    ),
                }
                for message in messages
            ],
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
        try:
            parsed = json.loads(value["message"]["content"])
            parsed.update(
                {
                    "input_tokens": value.get("prompt_eval_count"),
                    "output_tokens": value.get("eval_count"),
                    "measurements": {
                        key: value[key] / 1e9
                        for key in (
                            "load_duration",
                            "prompt_eval_duration",
                            "eval_duration",
                        )
                        if key in value
                    },
                }
            )
            return Reply.parse(parsed)
        except (KeyError, ValueError, TypeError) as error:
            raise RuntimeFailure("Malformed backend response") from error

    async def complete(self, messages, config):
        # Socket work is finite. Cancellation discards the result; no tool executes.
        return await asyncio.to_thread(self._request, messages, config)
