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


from recommender.models import Problem, Submission
from featurize import build_feature_matrix
from user_profile import build_user_profile


def recommend_for_user(k=10):

    # Get all problem vectors
    X, problems, mlb = build_feature_matrix()

    # Build current user's profile
    user_vector = build_user_profile()

    if user_vector is None:
        return []

    # Train KNN on all problems
    model = NearestNeighbors(
        n_neighbors=len(problems),
        metric="cosine"
    )

    model.fit(X)

    # Find nearest problems to user profile
    distances, indices = model.kneighbors(
        user_vector.reshape(1, -1)
    )

    # Get solved problem IDs
    solved_ids = set(
        Submission.objects.values_list(
            "problem_id",
            flat=True
        )
    )

    recommendations = []

    for distance, index in zip(
        distances[0],
        indices[0]
    ):

        problem = problems[index]

        # Don't recommend already solved problems
        if problem.id in solved_ids:
            continue

        recommendations.append({
            "problem": problem,
            "distance": float(distance),
            "similarity": 1 - float(distance)
        })

        if len(recommendations) >= k:
            break

    return recommendations


if __name__ == "__main__":

    recommendations = recommend_for_user(k=10)

    print("\nRecommended problems:\n")

    for rank, result in enumerate(
        recommendations,
        start=1
    ):

        problem = result["problem"]

        print(
            f"{rank}. "
            f"{problem.leetcode_id}. "
            f"{problem.title} | "
            f"{problem.difficulty} | "
            f"similarity={result['similarity']:.4f}"
        )