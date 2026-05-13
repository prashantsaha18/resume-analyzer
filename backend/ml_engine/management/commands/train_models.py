import time
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = "Train all ML models on synthetic resume data"
    def add_arguments(self, parser):
        parser.add_argument("--samples", type=int, default=5000)
        parser.add_argument("--quick",   action="store_true")
    def handle(self, *args, **options):
        n = 1000 if options["quick"] else options["samples"]
        self.stdout.write(self.style.WARNING(f"\n Training ML models on {n} samples...\n"))
        t0 = time.time()
        try:
            from ml_engine.trainer import train_all_models
            train_all_models(n_samples=n)
            elapsed = time.time() - t0
            self.stdout.write(self.style.SUCCESS(
                f"\n Training complete in {elapsed:.1f}s\n"
                "  ML models saved to backend/ml_engine/models/\n"
                "  Add GEMINI_API_KEY to .env anytime to use Gemini instead.\n"
            ))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"\n Training failed: {e}"))
            raise