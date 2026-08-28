import json
from pathlib import Path

from django.core.management.base import BaseCommand
from recommender.models import Problem


class Command(BaseCommand):
    help = "Import LeetCode problems from problems.json"

    def handle(self, *args, **options):

        file_path = (
            Path(__file__).resolve().parents[4]
            / "data"
            / "problems.json"
        )

        if not file_path.exists():
            self.stdout.write(
                self.style.ERROR(f"File not found: {file_path}")
            )
            return

        with open(file_path, "r", encoding="utf-8") as file:
            problems = json.load(file)

        created = 0
        updated = 0

        for data in problems:

            topics = [
                topic["name"]
                for topic in data.get("topicTags", [])
            ]

            _, was_created = Problem.objects.update_or_create(
                leetcode_id=int(data["frontendQuestionId"]),
                defaults={
                    "title": data["title"],
                    "slug": data["titleSlug"],
                    "difficulty": data["difficulty"],
                    "acceptance_rate": data.get("acRate"),
                    "topics": topics,
                },
            )

            if was_created:
                created += 1
            else:
                updated += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Import complete: {created} created, {updated} updated."
            )
        )