import os

# Base defaults (seconds)
_DEFAULT_TIMEOUT = 300
_DEFAULT_KEEPALIVE = 5

# Optional env overrides (if unset, use defaults above)
timeout = int(os.getenv("GUNICORN_TIMEOUT", str(_DEFAULT_TIMEOUT)))
graceful_timeout = timeout
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", str(_DEFAULT_KEEPALIVE)))

# Required by task
workers = 1
worker_class = "sync"

# Prove config is loaded on boot
print(
    f"GUNICORN_CONFIG_LOADED timeout={timeout} "
    f"graceful_timeout={graceful_timeout} "
    f"keepalive={keepalive} workers={workers} "
    f"worker_class={worker_class}"
)

