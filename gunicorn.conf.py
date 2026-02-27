import os

# Base defaults (seconds)
_DEFAULT_TIMEOUT = 300
_DEFAULT_KEEPALIVE = 5

# Optional env overrides
_timeout = int(os.getenv("GUNICORN_TIMEOUT", str(_DEFAULT_TIMEOUT)))
_keepalive = int(os.getenv("GUNICORN_KEEPALIVE", str(_DEFAULT_KEEPALIVE)))

# Top-level Gunicorn settings (required by task)
timeout = _timeout
graceful_timeout = _timeout
keepalive = _keepalive
workers = 1
worker_class = "sync"

print(f"GUNICORN_CONFIG_LOADED timeout={timeout} graceful_timeout={graceful_timeout} keepalive={keepalive} workers={workers}")

# Gunicorn config: allow long-running requests (e.g. Step0 o3-pro ~4 min)
# Prevents worker SIGTERM during /api/preview; timeout handled by our OpenAI/httpx timeouts instead.
import os

# Worker killed after this many seconds of no progress (default 30); 300 = 5 min
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "300"))
# Time to wait for workers to finish on graceful restart
graceful_timeout = int(os.environ.get("GUNICORN_GRACEFUL_TIMEOUT", "300"))
# Keep-alive to client (seconds)
keepalive = int(os.environ.get("GUNICORN_KEEPALIVE", "5"))

# Optional: bind (Render sets PORT)
# bind = "0.0.0.0:8000"
# workers: leave default unless needed
