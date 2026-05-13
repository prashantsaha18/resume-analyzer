"""
Vercel WSGI entry point for Django.
On Vercel, Firebase credentials come from FIREBASE_CREDENTIALS_JSON env var
(not a file), so we write them to /tmp at startup.
"""
import os
import json
import tempfile

# ── Write Firebase credentials from env var to /tmp ───────────────────────────
_firebase_json = os.environ.get("FIREBASE_CREDENTIALS_JSON", "")
if _firebase_json and not os.path.exists("/tmp/firebase-credentials.json"):
    try:
        creds = json.loads(_firebase_json)
        with open("/tmp/firebase-credentials.json", "w") as f:
            json.dump(creds, f)
        os.environ.setdefault("FIREBASE_CREDENTIALS_PATH", "/tmp/firebase-credentials.json")
    except Exception as e:
        print(f"Warning: Could not write Firebase credentials: {e}")

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "resumeai.settings_vercel")
os.environ.setdefault("FIREBASE_CREDENTIALS_PATH", "/tmp/firebase-credentials.json")

from django.core.wsgi import get_wsgi_application
app = get_wsgi_application()