# LeetCode Problem Recommender

An ML-based system that recommends the **next LeetCode problem to solve** based on a user's solved problems and problem-solving patterns.

The system uses **K-Nearest Neighbors (KNN)** to find similar problems and generate personalized recommendations.

## Features

* Fetch LeetCode problem and submission data
* Track solved problems
* Analyze topics and difficulty
* Recommend similar unsolved problems
* Personalized recommendations based on solving history
* Topic-wise solving statistics
* "Why this?" explanation for recommendations
* React dashboard

## How It Works

```text
LeetCode Data
      ↓
User Solving History
      ↓
Feature Extraction
      ↓
KNN Recommendation
      ↓
Unsolved Similar Problems
      ↓
Next Problem to Solve
```

## Tech Stack

* **Frontend:** React
* **Backend:** Django + Django REST Framework
* **Database:** SQLite / PostgreSQL
* **ML:** Python + Scikit-learn
* **Algorithm:** K-Nearest Neighbors
* **Data:** LeetCode GraphQL

## Project Structure

```text
leetcode-recommender/
│
├── data/          # LeetCode data collection
├── ml/            # Feature engineering & KNN
├── backend/       # Django REST API
├── frontend/      # React application
│
├── requirements.txt
└── README.md
```

## Recommendation Modes

### Similar Problems

Select a problem and get similar unsolved problems.

### For You

Get recommendations based on your recent solving history, difficulty, and topics.

## API

```text
GET /api/recommend/for-me/
GET /api/recommend/similar/<problem_id>/
GET /api/problems/
GET /api/problems/<problem_id>/
GET /api/stats/
```

## Development Roadmap

* [ ] LeetCode data collection
* [ ] Django database setup
* [ ] Feature engineering
* [ ] KNN recommendation model
* [ ] Django REST API
* [ ] React dashboard
* [ ] Personalized recommendations
* [ ] Deployment
* [ ] Automated data/model updates

## Goal

Build a personalized LeetCode practice assistant that learns from your solving history and continuously recommends what you should solve next.


