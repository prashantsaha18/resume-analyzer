from django.core.management.base import BaseCommand
from users.models import User
class Command(BaseCommand):
    help = "Create a demo user for development"
    def handle(self, *args, **options):
        user, created = User.objects.get_or_create(
            firebase_uid="demo_uid_dev_12345",
            defaults={"email": "demo@resumeai.dev", "display_name": "Demo User", "plan": "pro"}
        )
        self.stdout.write(self.style.SUCCESS(f"Demo user: {user.email} (created={created})"))