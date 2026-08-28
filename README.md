LeetCode Problem Recommender

A personalized LeetCode problem recommendation system using KNN. It analyzes solved problems, topics, difficulty, and acceptance rate to recommend relevant unsolved problems.

Current Stack

Backend: Django

Database: SQLite

ML: Python, NumPy, scikit-learn, joblib

Frontend: React

Data: LeetCode GraphQL API

How It Works

LeetCode
   ↓
Problems + submissions
   ↓
Django / SQLite
   ↓
Feature vectors
   ↓
KNN
   ↓
Personalized user profile
   ↓
Similarity + difficulty scoring
   ↓
Top unsolved problems

Current ML Pipeline

Each problem is represented using:

Topic tags

Difficulty

Acceptance rate

The current dataset contains approximately 4,000+ problems and the feature matrix is generated automatically.

The user profile is built from solved problems using recency-weighted averaging, giving newer solves more influence.

Recommendations are then ranked using:

80% similarity
20% difficulty suitability

Already-solved problems are filtered out.

Project Structure

leetcode_rec_sys/
│
├── data/
│   ├── fetch_lc.py
│   ├── problems.json
│   └── submissions.json
│
├── ml/
│   ├── featurize.py
│   ├── user_profile.py
│   ├── knn_model.py
│   ├── recommend.py
│   └── models/
│       ├── knn.pkl
│       ├── topic_encoder.pkl
│       └── features.npy
│
├── backend/
│   ├── manage.py
│   ├── config/
│   └── recommender/
│       ├── models.py
│       ├── views.py
│       └── urls.py
│
└── frontend/
    └── React application

Django API

Current endpoint:

GET /api/recommend/for-me/

Returns the user's top recommended unsolved problems.

Example:

{
  "count": 10,
  "recommendations": [
    {
      "leetcode_id": 46,
      "title": "Permutations",
      "difficulty": "Medium",
      "similarity": 0.86,
      "score": 0.89
    }
  ]
}

Running Locally

1. Install dependencies

pip install django numpy scikit-learn joblib requests

2. Fetch LeetCode data

From the project root:

python data/fetch_lc.py

3. Set up Django database

cd backend
python manage.py makemigrations
python manage.py migrate
python manage.py import_problems
python manage.py import_submissions

4. Train the KNN model

From the project root:

python ml/knn_model.py

5. Start Django

cd backend
python manage.py runserver

API:

http://127.0.0.1:8000/api/recommend/for-me/

Current Status

LeetCode problem data fetching

Problem database import

Submission database import

Feature engineering

KNN model

Recency-weighted user profile

Difficulty-aware ranking

Personalized recommendation API

React dashboard

Similar-problem API

Topic statistics API

Recommendation explanations

Deployment

Automated data refresh

V2 Ideas

Topic diversity in recommendations

Better historical submission data

Solve time / attempts

Problem statement embeddings

Hybrid metadata + embedding similarity

Company-based filtering

Recommendation feedback loop

Scheduled model/data refresh

Note

The current version is intentionally simple and explainable. The ML model uses engineered metadata features rather than semantic embeddings.
