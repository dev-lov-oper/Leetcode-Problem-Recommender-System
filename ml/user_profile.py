from featurize import build_feature_matrix
import os
import sys
import numpy as np
import django


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

def build_user_profile():

    X, problems, mlb = build_feature_matrix()

    submissions = get_solved_submissions()

    if not submissions:
        return None

    # Map Django problem ID → feature-vector index
    problem_index = {
        problem.id: index
        for index, problem in enumerate(problems)
    }

    weighted_vectors = []
    weights = []

    for rank, submission in enumerate(submissions):

        problem_id = submission.problem.id

        if problem_id not in problem_index:
            continue

        index = problem_index[problem_id]

        vector = X[index]

        # Newer problems get higher weight
        weight = 1 / (rank + 1)

        weighted_vectors.append(vector * weight)
        weights.append(weight)

    if not weighted_vectors:
        return None

    user_vector = (
        np.sum(weighted_vectors, axis=0)
        / np.sum(weights)
    )

    return user_vector

if __name__ == "__main__":

    user_vector = build_user_profile()

    if user_vector is not None:
        print("User vector shape:", user_vector.shape)

        print("\nFirst 20 values:")
        print(user_vector[:20])

        print("\nMinimum:", user_vector.min())
        print("Maximum:", user_vector.max())