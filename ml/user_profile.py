from featurize import build_feature_matrix
import os
import sys
import numpy as np
import django
from collections import Counter

# -----------------------------
# Django setup
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")

sys.path.append(BACKEND_DIR)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

# pyrefly: ignore [missing-import]
from recommender.models import Problem, Submission


def get_solved_submissions():
    submissions = (
        Submission.objects
        .select_related("problem")
        .order_by("-submitted_at")
    )

    # Keep only the latest submission for each problem
    latest = {}

    for submission in submissions:
        problem_id = submission.problem.id

        if problem_id not in latest:
            latest[problem_id] = submission

    return list(latest.values())

def get_solved_vectors():

    X, problems, mlb = build_feature_matrix()

    solved_submissions = get_solved_submissions()

    solved_ids = {
        submission.problem.id
        for submission in solved_submissions
    }

    solved_vectors = []

    for index, problem in enumerate(problems):

        if problem.id in solved_ids:
            solved_vectors.append(X[index])

    return solved_vectors

def build_user_profile_from_slugs(solved_slugs, X, problems):
    """
    Builds a weighted user feature vector from a list of solved problem titleSlugs.
    Recency decay weight = 1 / (rank + 1).
    """
    if not solved_slugs:
        return None

    # Map problem titleSlug -> index in feature matrix
    slug_to_index = {
        problem.slug: index
        for index, problem in enumerate(problems)
    }

    weighted_vectors = []
    weights = []

    # Filter unique slugs preserving recency order
    seen = set()
    unique_slugs = []
    for slug in solved_slugs:
        if slug not in seen:
            seen.add(slug)
            unique_slugs.append(slug)

    for rank, slug in enumerate(unique_slugs):
        if slug not in slug_to_index:
            continue
        index = slug_to_index[slug]
        vector = X[index]
        weight = 1.0 / (rank + 1)
        weighted_vectors.append(vector * weight)
        weights.append(weight)

    if not weighted_vectors:
        return None

    user_vector = np.sum(weighted_vectors, axis=0) / np.sum(weights)
    return user_vector


def get_difficulty_profile_from_slugs(solved_slugs, problems):
    """
    Returns a Counter of problem difficulties for solved problem titleSlugs.
    """
    slug_to_diff = {
        problem.slug: problem.difficulty
        for problem in problems
    }
    difficulties = [
        slug_to_diff[slug]
        for slug in solved_slugs
        if slug in slug_to_diff
    ]
    return Counter(difficulties)


def get_target_difficulty_from_profile(diff_counter):
    if not diff_counter:
        return "Easy"
    # If user has solved equal or more Mediums/Hards, bump target difficulty appropriately
    most_common = diff_counter.most_common()
    if most_common:
        return most_common[0][0]
    return "Easy"


def build_user_profile():
    X, problems, mlb = build_feature_matrix()
    submissions = get_solved_submissions()
    if not submissions:
        return None
    solved_slugs = [s.problem.slug for s in submissions]
    return build_user_profile_from_slugs(solved_slugs, X, problems)


def get_difficulty_profile():
    X, problems, mlb = build_feature_matrix()
    submissions = get_solved_submissions()
    solved_slugs = [s.problem.slug for s in submissions]
    return get_difficulty_profile_from_slugs(solved_slugs, problems)


def get_target_difficulty():
    profile = get_difficulty_profile()
    return get_target_difficulty_from_profile(profile)


if __name__ == "__main__":
    profile = get_difficulty_profile()
    print("Difficulty profile:")
    print(profile)
    print("Target difficulty:", get_target_difficulty())

