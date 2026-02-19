# mini-aegaeon

Rough skeleton for Aegaeon with:

- Proxy routing against synchronized metadata
- OpenAI-compatible proxy endpoints for request intake
- Prefill/decode scheduling hooks and batch runner boundary
- VRAM/model-cache/KV memory management primitives

## Metadata backends

- `PosixShmMetadataStore`: single-host POSIX shared memory backend
- `RedisMetadataStore`: Redis backend

Both implement the same `MetadataStore` interface.

## Proxy endpoint

Run:

```bash
python -m aegaeon --host 0.0.0.0 --port 8080
```

Routes:

- `POST /v1/chat/completions`
- `POST /v1/completions`

Admin metadata routes:

- `POST /admin/register_instance`
- `POST /admin/update_status`

Routing policy (current minimal version):

1. Proxy calls `sync_for_routing()` before each request.
2. Prefer `PREFILL` instances that already serve `model`.
3. If none serve it, enqueue to shortest-queue `PREFILL` instance.
