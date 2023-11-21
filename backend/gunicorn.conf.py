# Gunicorn configuration file
import os
import multiprocessing

max_requests = 1000
max_requests_jitter = 50

log_file = "-"

port = os.environ.get("BACKEND_PORT", "8080")

bind = f"0.0.0.0:{port}"

worker_class = "uvicorn.workers.UvicornWorker"
workers = (multiprocessing.cpu_count() * 2) + 1
