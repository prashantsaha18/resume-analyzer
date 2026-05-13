import logging
import firebase_admin
from firebase_admin import auth, credentials
from django.conf import settings
from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed
from users.models import User

logger = logging.getLogger(__name__)
_firebase_initialized = False

def init_firebase():
    global _firebase_initialized
    if not _firebase_initialized:
        try:
            cred = credentials.Certificate(settings.FIREBASE_CREDENTIALS_PATH)
            firebase_admin.initialize_app(cred)
            _firebase_initialized = True
        except Exception as e:
            logger.error(f"Firebase init failed: {e}")

class FirebaseAuthentication(BaseAuthentication):
    def authenticate(self, request):
        header = request.META.get("HTTP_AUTHORIZATION", "")
        if not header.startswith("Bearer "):
            return None
        token = header.split("Bearer ")[1].strip()
        if not token:
            return None
        try:
            init_firebase()
            decoded = auth.verify_id_token(token)
            uid   = decoded["uid"]
            email = decoded.get("email", "")
            name  = decoded.get("name", "")
            photo = decoded.get("picture", "")
            user, created = User.objects.get_or_create(
                firebase_uid=uid,
                defaults={"email": email, "display_name": name, "photo_url": photo}
            )
            if not created:
                changed = False
                if email and user.email != email: user.email = email; changed = True
                if name  and user.display_name != name: user.display_name = name; changed = True
                if changed: user.save()
            return (user, token)
        except auth.InvalidIdTokenError:
            raise AuthenticationFailed("Invalid Firebase token.")
        except auth.ExpiredIdTokenError:
            raise AuthenticationFailed("Firebase token has expired.")
        except Exception as e:
            logger.error(f"Firebase auth error: {e}")
            raise AuthenticationFailed("Authentication failed.")
    def authenticate_header(self, request):
        return "Bearer"