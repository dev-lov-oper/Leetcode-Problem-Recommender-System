# LeetCode Problem Recommender

> A personalized recommendation system that uses **KNN-based similarity** to recommend the next LeetCode problems you should solve.

## 🚀 Overview

The system analyzes your LeetCode solving history and recommends **unsolved problems** based on:

* 🏷️ Problem topics
* 📊 Difficulty
* 📈 Acceptance rate
* 🕒 Recent solving activity
* 🤖 KNN-based similarity

The goal is to make problem selection personalized instead of randomly choosing problems.

---

## 🧠 How It Works

```text
            LeetCode
                │
                ▼
       Problems + Submissions
                │
                ▼
          Django + SQLite
                │
                ▼
        Feature Engineering
                │
                ▼
        Problem Feature Vectors
                │
                ▼
              KNN
                │
                ▼
       Personalized User Profile
                │
                ▼
    Similarity + Difficulty Score
                │
                ▼
        Top Unsolved Problems
```

---

## 🛠️ Tech Stack

| Component        | Technology           |
| ---------------- | -------------------- |
| Backend          | Django               |
| Database         | SQLite               |
| Machine Learning | Scikit-learn         |
| Data Processing  | NumPy                |
| Model Storage    | Joblib               |
| Frontend         | React                |
| Data Source      | LeetCode GraphQL API |

---

## 📂 Project Structure

```text
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
```

---

## 📊 ML Pipeline

Each LeetCode problem is converted into a numerical **feature vector** using:

* 🏷️ Topic tags
* 🎯 Difficulty
* 📈 Acceptance rate

The current dataset contains **4,000+ LeetCode problems**.

The user's profile is generated from their solved problems using **recency-weighted averaging**, so recent solving activity has greater influence.

### Recommendation Score

```text
Final Score =
    80% × Similarity
    +
    20% × Difficulty Suitability
```

Already-solved problems are automatically excluded.

---

## 🔌 API

### Get Personalized Recommendations

```http
GET /api/recommend/for-me/
```

Example response:

```json
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
```

---

## ⚙️ Setup

### 1. Install Dependencies

```bash
pip install django numpy scikit-learn joblib requests
```

### 2. Fetch LeetCode Data

From the project root:

```bash
python data/fetch_lc.py
```

### 3. Setup Database

```bash
cd backend

python manage.py makemigrations
python manage.py migrate

python manage.py import_problems
python manage.py import_submissions
```

### 4. Train KNN

From the project root:

```bash
python ml/knn_model.py
```

### 5. Start Django

```bash
cd backend
python manage.py runserver
```

API will be available at:

```text
http://127.0.0.1:8000/api/recommend/for-me/
```

---

## ✅ Current Progress

* [x] LeetCode problem data fetching
* [x] Problem database
* [x] Submission database
* [x] Feature engineering
* [x] Problem vectors
* [x] KNN model
* [x] Recency-weighted user profile
* [x] Difficulty-aware recommendations
* [x] Personalized recommendation API
* [ ] React dashboard
* [ ] Similar-problem API
* [ ] Topic statistics
* [ ] Recommendation explanations
* [ ] Deployment
* [ ] Automated data refresh

---

## 🔮 Future Improvements — V2

* 🎯 Topic diversity
* 🧩 Better historical submission tracking
* ⏱️ Solve time and attempt-based features
* 🧠 Problem-statement embeddings
* 🔀 Hybrid metadata + embedding similarity
* 🏢 Company-based filtering
* 📈 Recommendation feedback loop
* 🔄 Automated model retraining

---

## 🎯 Project Goal

Build a practical recommendation system that answers:

> **"I've solved these problems. What should I solve next?"**

The current version focuses on a **simple, explainable KNN approach** using engineered problem metadata rather than semantic embeddings.
