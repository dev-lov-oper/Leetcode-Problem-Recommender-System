import json
from pathlib import Path
from datetime import datetime, timezone

from django.core.management.base import BaseCommand

from recommender.models import Problem, Submission


class Command(BaseCommand):
    help = "Import LeetCode submissions"

    def handle(self, *args, **options):

        file_path = (
            Path(__file__).resolve().parents[4]
            / "data"
            / "submissions.json"
        )

        if not file_path.exists():
            self.stdout.write(
                self.style.ERROR(f"File not found: {file_path}")
            )
            return

        with open(file_path, "r", encoding="utf-8") as file:
            submissions = json.load(file)

        created = 0
        skipped = 0

        for data in submissions:

            try:
                problem = Problem.objects.get(
                    slug=data["titleSlug"]
                )
            except Problem.DoesNotExist:
                self.stdout.write(
                    self.style.WARNING(
                        f"Problem not found: {data['title']}"
                    )
                )
                skipped += 1
                continue

            submitted_at = datetime.fromtimestamp(
                int(data["timestamp"]),
                tz=timezone.utc
            )

            _, was_created = Submission.objects.update_or_create(
                submission_id=data["id"],
                defaults={
                    "problem": problem,
                    "title": data["title"],
                    "submitted_at": submitted_at,
                },
            )

            if was_created:
                created += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Import complete: {created} created, {skipped} skipped."
            )
        )