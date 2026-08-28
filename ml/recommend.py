import os
import sys

import django
import joblib

from sklearn.neighbors import NearestNeighbors


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
from featurize import build_feature_matrix
from user_profile import (
    build_user_profile_from_slugs,
    get_difficulty_profile_from_slugs,
    get_target_difficulty_from_profile
)

# Global in-memory cache for fast lookup
_CACHE = {
    "X": None,
    "problems": None,
    "mlb": None,
    "model": None,
    "slug_map": None
}


def get_cached_artifacts():
    if _CACHE["X"] is None:
        X, problems, mlb = build_feature_matrix()
        model = NearestNeighbors(n_neighbors=min(500, len(problems)), metric="cosine")
        model.fit(X)
        slug_map = {p.slug: p for p in problems}
        _CACHE["X"] = X
        _CACHE["problems"] = problems
        _CACHE["mlb"] = mlb
        _CACHE["model"] = model
        _CACHE["slug_map"] = slug_map
    return _CACHE["X"], _CACHE["problems"], _CACHE["mlb"], _CACHE["model"], _CACHE["slug_map"]


def recommend_for_solved_slugs(solved_slugs, k=10, filter_difficulty=None):
    X, problems, mlb, model, slug_map = get_cached_artifacts()

    solved_set = set(solved_slugs or [])
    diff_counter = get_difficulty_profile_from_slugs(solved_slugs or [], problems)
    target_difficulty = filter_difficulty or get_target_difficulty_from_profile(diff_counter)

    user_vector = build_user_profile_from_slugs(solved_slugs or [], X, problems)

    # Cold start fallback if user has no solved problems or no matching vectors
    if user_vector is None:
        # Default cold start: recommend high-acceptance Easy/Medium problems across varied topics
        candidates = []
        for problem in problems:
            if problem.slug in solved_set:
                continue
            if filter_difficulty and problem.difficulty != filter_difficulty:
                continue
            candidates.append({
                "problem": problem,
                "similarity": 0.5,
                "difficulty_score": 1.0 if problem.difficulty == target_difficulty else 0.5,
                "final_score": (problem.acceptance_rate or 50.0) / 100.0
            })
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        return candidates[:k], target_difficulty

    distances, indices = model.kneighbors(user_vector.reshape(1, -1))

    candidates = []
    for distance, index in zip(distances[0], indices[0]):
        problem = problems[index]

        if problem.slug in solved_set:
            continue

        if filter_difficulty and problem.difficulty != filter_difficulty:
            continue

        similarity = 1.0 - float(distance)
        diff_match = difficulty_score(problem.difficulty, target_difficulty)

        final_score = (0.75 * similarity) + (0.25 * diff_match)

        candidates.append({
            "problem": problem,
            "similarity": similarity,
            "difficulty_score": diff_match,
            "final_score": final_score
        })

    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    return candidates[:k], target_difficulty


def recommend_for_user(k=10):
    submissions = list(Submission.objects.all())
    solved_slugs = [s.problem.slug for s in submissions if hasattr(s, "problem") and s.problem]
    recs, _ = recommend_for_solved_slugs(solved_slugs, k=k)
    return recs


def difficulty_score(problem_difficulty, target_difficulty):
    levels = {"Easy": 1, "Medium": 2, "Hard": 3}
    problem_level = levels.get(problem_difficulty)
    target_level = levels.get(target_difficulty)
    if problem_level is None or target_level is None:
        return 0.0
    diff = abs(problem_level - target_level)
    if diff == 0:
        return 1.0
    elif diff == 1:
        return 0.5
    return 0.0


if __name__ == "__main__":
    recommendations = recommend_for_user(k=10)
    print(f"\nTop {len(recommendations)} Recommendations:")
    for rank, res in enumerate(recommendations, 1):
        p = res["problem"]
        print(f"{rank}. {p.leetcode_id}. {p.title} ({p.difficulty}) - Score: {res['final_score']:.3f}")