import os
import sys
import joblib
import numpy as np

from sklearn.neighbors import NearestNeighbors


# -----------------------------
# Django setup
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")

sys.path.append(BACKEND_DIR)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django
django.setup()

from recommender.models import Problem, Submission

from featurize import build_feature_matrix


# -----------------------------
# Train KNN
# -----------------------------

def train_knn():

    X, problems, mlb = build_feature_matrix()

    model = NearestNeighbors(
        n_neighbors=20,
        metric="cosine"
    )

    model.fit(X)

    os.makedirs(
        os.path.join(BASE_DIR, "ml", "models"),
        exist_ok=True
    )

    joblib.dump(
        model,
        os.path.join(BASE_DIR, "ml", "models", "knn.pkl")
    )

    joblib.dump(
        mlb,
        os.path.join(BASE_DIR, "ml", "models", "topic_encoder.pkl")
    )

    np.save(
        os.path.join(BASE_DIR, "ml", "models", "features.npy"),
        X
    )

    print("KNN model trained.")
    print("Problems:", len(problems))
    print("Features:", X.shape[1])

def find_similar(problem_id, k=5):

    X, problems, mlb = build_feature_matrix()

    model = joblib.load(
        os.path.join(BASE_DIR, "ml", "models", "knn.pkl")
    )

    # Find index of requested problem
    problem_index = next(
        i for i, p in enumerate(problems)
        if p.id == problem_id
    )

    distances, indices = model.kneighbors(
        X[problem_index].reshape(1, -1),
        n_neighbors=k + 1
    )

    results = []

    for distance, index in zip(
        distances[0][1:],
        indices[0][1:]
    ):
        results.append({
            "problem": problems[index],
            "distance": float(distance)
        })

    return results

if __name__ == "__main__":

    train_knn()

    results = find_similar(1, k=5)

    for result in results:
        p = result["problem"]

        print(
        f"{p.leetcode_id}. {p.title} | "
        f"{p.difficulty} | "
        f"{p.topics} | "
        f"distance={result['distance']:.4f}"
    )