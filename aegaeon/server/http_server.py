from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from uuid import uuid4

from aegaeon.runtime import InstanceRuntime
from aegaeon.types import Request

from aegaeon.proxy.metadata_store import InstanceStatus, RequestPhase
from aegaeon.proxy.proxy import Proxy
from aegaeon.proxy.router import RequestEnvelope


class OpenAIProxyHandler(BaseHTTPRequestHandler):
    proxy: Proxy
    local_instances: dict[str, InstanceRuntime]

    def _read_json(self) -> dict:
        content_len = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_len) if content_len > 0 else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def _write_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _print_route(self, request_id: str, model: str, instance_id: str, queued: bool) -> None:
        print(f"[route] req={request_id} model={model} instance={instance_id} queued={queued}")

    def do_POST(self) -> None:  # noqa: N802
        if self.path in {"/v1/chat/completions", "/v1/completions"}:
            self._handle_openai_completion()
            return
        self._write_json(404, {"error": {"message": f"unknown path: {self.path}"}})

    def _handle_openai_completion(self) -> None:
        try:
            payload = self._read_json()
        except Exception as exc:
            self._write_json(400, {"error": {"message": f"invalid json: {exc}"}})
            return

        model = payload.get("model")
        if not isinstance(model, str) or not model:
            self._write_json(400, {"error": {"message": "`model` is required"}})
            return

        request_id = self.headers.get("x-request-id") or f"req-{uuid4().hex}"
        req = RequestEnvelope(
            request_id=request_id,
            model=model,
            payload=payload,
            phase=RequestPhase.PREFILL,
        )

        try:
            decision = self.proxy.route(req)
            rt = self.local_instances.get(decision.instance_id)
            if rt is None:
                raise KeyError(f"local instance {decision.instance_id} not found")
            current_status = self.proxy.sync_instance_metadata().instances.get(decision.instance_id)
            rt.ensure_model_ready(model)

            prompt = payload.get("prompt")
            if prompt is None and isinstance(payload.get("messages"), list):
                prompt = " ".join(
                    str(m.get("content", "")) for m in payload["messages"] if isinstance(m, dict)
                )
            prompt_text = str(prompt or "")
            input_ids = [ord(c) % 256 for c in prompt_text][:128] or [0]

            rt.submit(
                Request(
                    request_id=request_id,
                    model=model,
                    input_ids=input_ids,
                    uid=int(time.time_ns() & 0x7FFFFFFF),
                    sampling_params={},
                )
            )
            result = rt.step()
            queue_after = max((current_status.queue_depth if current_status is not None else 1) - 1, 0)
            self.proxy.update_instance_status(
                decision.instance_id,
                InstanceStatus(
                    current_models=rt.requested_models(),
                    queue_depth=queue_after,
                ),
            )
        except Exception as exc:
            self._write_json(503, {"error": {"message": str(exc)}})
            return

        self._print_route(
            request_id=request_id,
            model=model,
            instance_id=decision.instance_id,
            queued=decision.queued,
        )

        text = "ok"
        if result is not None:
            out = result.outputs.get(request_id)
            if isinstance(out, dict) and "tokens" in out:
                text = f"tokens={out['tokens']}"

        response = {
            "id": request_id,
            "object": "chat.completion" if self.path.endswith("chat/completions") else "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "text": text,
                    "finish_reason": "stop",
                }
            ],
        }
        self._write_json(200, response)


class OpenAIProxyServer(ThreadingHTTPServer):
    def __init__(
        self,
        host: str,
        port: int,
        proxy: Proxy,
        local_instances: dict[str, InstanceRuntime] | None = None,
    ) -> None:
        handler = type(
            "BoundOpenAIProxyHandler",
            (OpenAIProxyHandler,),
            {"proxy": proxy, "local_instances": local_instances or {}},
        )
        super().__init__((host, port), handler)


def run_proxy_server(
    proxy: Proxy,
    host: str = "0.0.0.0",
    port: int = 8080,
    local_instances: dict[str, InstanceRuntime] | None = None,
) -> None:
    server = OpenAIProxyServer(host=host, port=port, proxy=proxy, local_instances=local_instances)
    try:
        server.serve_forever()
    finally:
        server.server_close()
