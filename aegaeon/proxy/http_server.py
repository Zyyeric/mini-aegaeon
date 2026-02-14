from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from uuid import uuid4

from .metadata_store import InstanceInfo, InstancePhase, InstanceRole, InstanceStatus, RequestPhase
from .router import RequestEnvelope
from .service import ProxyLayer


class OpenAIProxyHandler(BaseHTTPRequestHandler):
    proxy: ProxyLayer

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
        if self.path == "/admin/register_instance":
            self._handle_register_instance()
            return
        if self.path == "/admin/update_status":
            self._handle_update_status()
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
        except Exception as exc:
            self._write_json(503, {"error": {"message": str(exc)}})
            return

        self._print_route(
            request_id=request_id,
            model=model,
            instance_id=decision.instance_id,
            queued=decision.queued,
        )

        response = {
            "id": request_id,
            "object": "request.accepted",
            "created": int(time.time()),
            "model": model,
            "status": "accepted",
        }
        self._write_json(202, response)

    def _handle_register_instance(self) -> None:
        try:
            payload = self._read_json()
            instance_id = str(payload["instance_id"])
            role = InstanceRole(payload["role"])
            endpoint = str(payload["endpoint"])
        except Exception as exc:
            self._write_json(400, {"error": {"message": f"invalid payload: {exc}"}})
            return

        self.proxy.register_instance(
            info=InstanceInfo(
                instance_id=instance_id,
                role=role,
                endpoint=endpoint,
            ),
        )
        self._write_json(200, {"ok": True})

    def _handle_update_status(self) -> None:
        try:
            payload = self._read_json()
            instance_id = str(payload["instance_id"])
            models = set(str(m) for m in payload.get("current_models", []))
            phase = InstancePhase(payload["phase"])
            queue_depth = int(payload["queue_depth"])
        except Exception as exc:
            self._write_json(400, {"error": {"message": f"invalid payload: {exc}"}})
            return

        self.proxy.update_instance_status(
            instance_id=instance_id,
            status=InstanceStatus(current_models=models, phase=phase, queue_depth=queue_depth),
        )
        self._write_json(200, {"ok": True})


class OpenAIProxyServer(ThreadingHTTPServer):
    def __init__(self, host: str, port: int, proxy: ProxyLayer) -> None:
        handler = type("BoundOpenAIProxyHandler", (OpenAIProxyHandler,), {"proxy": proxy})
        super().__init__((host, port), handler)


def run_proxy_server(proxy: ProxyLayer, host: str = "0.0.0.0", port: int = 8080) -> None:
    server = OpenAIProxyServer(host=host, port=port, proxy=proxy)
    try:
        server.serve_forever()
    finally:
        server.server_close()
