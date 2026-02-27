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
