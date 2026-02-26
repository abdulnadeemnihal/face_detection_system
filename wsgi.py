"""
wsgi.py — WSGI entry point for production servers (gunicorn, waitress, etc.).
"""

from app import create_app

application = create_app()

if __name__ == '__main__':
    application.run()
