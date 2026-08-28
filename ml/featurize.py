import os
import sys

import django
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


# Tell Python where manage.py is
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")

sys.path.append(BACKEND_DIR)

# Initialize Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

# pyrefly: ignore [missing-import]
from recommender.models import Problem


def build_feature_matrix():

    problems = list(Problem.objects.all())

    # -----------------------------
    # 1. Extract topics
    # -----------------------------

    topic_lists = [problem.topics for problem in problems]

    mlb = MultiLabelBinarizer()
    topic_matrix = mlb.fit_transform(topic_lists)

    # -----------------------------
    # 2. Difficulty
    # -----------------------------

    difficulty_map = {
        "Easy": 1,
        "Medium": 2,
        "Hard": 3
    }

    difficulty = np.array([
        difficulty_map.get(problem.difficulty, 0)
        for problem in problems
    ]).reshape(-1, 1)

    # Normalize difficulty to 0–1
    difficulty = difficulty / 3.0

    # -----------------------------
    # 3. Acceptance rate
    # -----------------------------

    acceptance = np.array([
        problem.acceptance_rate or 0
        for problem in problems
    ]).reshape(-1, 1)

    # Convert percentage to 0–1
    acceptance = acceptance / 100.0

    # -----------------------------
    # 4. Combine everything
    # -----------------------------

    X = np.hstack([
        topic_matrix,
        difficulty,
        acceptance
    ])

    return X, problems, mlb


if __name__ == "__main__":

    X, problems, mlb = build_feature_matrix()

    print("Number of problems:", len(problems))
    print("Number of features:", X.shape[1])
    print("Matrix shape:", X.shape)

    print("\nFirst problem:")
    print(problems[0].title)

    print("\nFirst vector:")
    print(X[0])

    print("\nNumber of topics:")
    print(len(mlb.classes_))

    print("\nTopics:")
    print(list(mlb.classes_))