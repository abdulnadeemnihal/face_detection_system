"""
gunicorn_config.py — Gunicorn WSGI server configuration.
"""

import os

bind: str = '0.0.0.0:5000'
workers: int = 2
threads: int = 4
timeout: int = 120
worker_class: str = 'gthread'
preload_app: bool = True

# Logging
os.makedirs('logs', exist_ok=True)
accesslog: str = 'logs/access.log'
errorlog: str = 'logs/error.log'
loglevel: str = 'info'
