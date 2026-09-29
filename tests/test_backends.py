import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agent_lab.backends import OllamaAdapter
from agent_lab.runtime import AgentConfig, RuntimeFailure


class BackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_protocol_and_sampling_capture(self):
        captured = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                captured.append(
                    json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                )
                payload = json.dumps(
                    {
                        "message": {"content": '{"content":"5","calls":[]}'},
                        "prompt_eval_count": 12,
                        "eval_count": 5,
                        "eval_duration": 100,
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_):
                pass  # The fixture intentionally has no request logging.

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                adapter = OllamaAdapter(f"http://127.0.0.1:{server.server_port}")
                reply = await adapter.complete(
                    [{"role": "user", "content": "sum"}],
                    AgentConfig(seed=42, max_output_tokens=17),
                )
                self.assertEqual(reply.content, "5")
                self.assertEqual(reply.output_tokens, 5)
                self.assertEqual(captured[0]["options"]["seed"], 42)
                self.assertEqual(captured[0]["options"]["num_predict"], 17)
                self.assertFalse(captured[0]["stream"])
            finally:
                server.shutdown()
                thread.join()

    def test_external_endpoints_and_bad_replies_are_rejected(self):
        from agent_lab.runtime import Reply

        with self.assertRaises(ValueError):
            OllamaAdapter("https://example.com")
        with self.assertRaises(RuntimeFailure):
            Reply.parse({"calls": [{"name": "sum", "arguments": []}]})
