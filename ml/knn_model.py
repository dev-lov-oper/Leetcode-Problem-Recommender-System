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


if __name__ == "__main__":
    train_knn()