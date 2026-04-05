import os

# Base defaults (seconds)
_DEFAULT_TIMEOUT = 300
_DEFAULT_KEEPALIVE = 5

# Optional env overrides (if unset, use defaults above)
timeout = int(os.getenv("GUNICORN_TIMEOUT", str(_DEFAULT_TIMEOUT)))
graceful_timeout = timeout
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", str(_DEFAULT_KEEPALIVE)))

# At least 2 sync workers so one can serve /health while another handles long /api/generate-video.
# Host env may set GUNICORN_WORKERS=1; clamp minimum to 2 so health checks are not starved.
_workers_raw = int(os.getenv("GUNICORN_WORKERS", "2"))
workers = max(2, _workers_raw)
worker_class = "sync"

# Prove config is loaded on boot
print(
    f"GUNICORN_CONFIG_LOADED timeout={timeout} "
    f"graceful_timeout={graceful_timeout} "
    f"keepalive={keepalive} workers={workers} "
    f"worker_class={worker_class}"
)

