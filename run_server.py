"""Production entry point for Cloud Run."""
import os
import sys

# Force unbuffered stdout so Cloud Run captures all logs immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from app import app, socketio

port = int(os.environ.get("PORT", 8080))
socketio.run(app, host="0.0.0.0", port=port, allow_unsafe_werkzeug=True)
