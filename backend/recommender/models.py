from django.db import models


class Problem(models.Model):
    leetcode_id = models.IntegerField(unique=True)
    title = models.CharField(max_length=255)
    slug = models.CharField(max_length=255, unique=True)
    difficulty = models.CharField(max_length=20)
    acceptance_rate = models.FloatField(null=True, blank=True)
    topics = models.JSONField(default=list)

    def __str__(self):
        return f"{self.leetcode_id}. {self.title}"