from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin

class UserManager(BaseUserManager):
    def create_user(self, firebase_uid, email, **extra):
        email = self.normalize_email(email)
        user = self.model(firebase_uid=firebase_uid, email=email, **extra)
        user.set_unusable_password()
        user.save(using=self._db)
        return user
    def create_superuser(self, firebase_uid, email, **extra):
        extra.setdefault("is_staff", True)
        extra.setdefault("is_superuser", True)
        return self.create_user(firebase_uid, email, **extra)

class User(AbstractBaseUser, PermissionsMixin):
    firebase_uid = models.CharField(max_length=128, unique=True, db_index=True)
    email        = models.EmailField(unique=True)
    display_name = models.CharField(max_length=150, blank=True)
    photo_url    = models.URLField(blank=True)
    is_active    = models.BooleanField(default=True)
    is_staff     = models.BooleanField(default=False)
    date_joined  = models.DateTimeField(auto_now_add=True)
    last_login   = models.DateTimeField(auto_now=True)
    plan         = models.CharField(max_length=20, default="free",
                    choices=[("free","Free"),("pro","Pro"),("enterprise","Enterprise")])
    credits_used = models.PositiveIntegerField(default=0)
    objects = UserManager()
    USERNAME_FIELD  = "firebase_uid"
    REQUIRED_FIELDS = ["email"]
    class Meta:
        db_table = "users"
    def __str__(self):
        return self.email
    @property
    def full_name(self):
        return self.display_name or self.email.split("@")[0]