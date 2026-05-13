"""
Vercel Production Settings
No sklearn/heavy ML libs. Uses rule-based inference or Gemini if key set.
"""
import os
from dotenv import load_dotenv
load_dotenv()

from resumeai.settings import *   # noqa

DEBUG      = False
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "fallback-change-me")

ALLOWED_HOSTS = [
    ".vercel.app",
    ".now.sh",
    "localhost",
    "127.0.0.1",
    os.getenv("VERCEL_URL", ""),
    *[h.strip() for h in os.getenv("ALLOWED_HOSTS", "").split(",") if h.strip()],
]

# WhiteNoise for static files
MIDDLEWARE = [MIDDLEWARE[0]] + ["whitenoise.middleware.WhiteNoiseMiddleware"] + MIDDLEWARE[1:]
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"
STATIC_ROOT = os.path.join(BASE_DIR, "staticfiles")
STATIC_URL  = "/static/"

# CORS
_cors_env = os.getenv("CORS_ALLOWED_ORIGINS", "https://resumeai-pro.vercel.app")
CORS_ALLOWED_ORIGINS = [u.strip() for u in _cors_env.split(",") if u.strip()]
CORS_ALLOW_ALL_ORIGINS = False

# Database (Neon, always SSL)
DATABASES = {
    "default": {
        "ENGINE":   "django.db.backends.postgresql",
        "NAME":     os.environ.get("DB_NAME", "neondb"),
        "USER":     os.environ.get("DB_USER", ""),
        "PASSWORD": os.environ.get("DB_PASSWORD", ""),
        "HOST":     os.environ.get("DB_HOST", ""),
        "PORT":     os.environ.get("DB_PORT", "5432"),
        "OPTIONS":  {"sslmode": "require"},
        "CONN_MAX_AGE": 60,
    }
}

# Firebase path (written to /tmp by api/index.py)
FIREBASE_CREDENTIALS_PATH = os.environ.get(
    "FIREBASE_CREDENTIALS_PATH", "/tmp/firebase-credentials.json"
)

# Gemini (optional)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# No background tasks on serverless
CELERY_TASK_ALWAYS_EAGER = True

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "root":     {"handlers": ["console"], "level": "WARNING"},
}