import os
import sys
from pathlib import Path
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_GET

# Add ml/ and data/ directories to python path
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

if str(PROJECT_ROOT / "ml") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "ml"))
if str(PROJECT_ROOT / "data") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "data"))

from fetch_lc import fetch_submissions, fetch_user_profile
from recommend import recommend_for_solved_slugs, get_cached_artifacts
from knn_model import find_similar
from recommender.models import Problem


def index_view(request):
    """Renders the main LeetCode Recommender SPA dashboard."""
    return render(request, "index.html")


@require_GET
def recommend_user_api(request):
    """
    API endpoint: GET /api/recommend/?username=xxx&k=10&difficulty=...
    Dynamically fetches user's profile & recent submissions from LeetCode GraphQL,
    computes feature matrix vectors, and returns personalized recommendations.
    """
    username = request.GET.get("username", "").strip()
    k_param = request.GET.get("k", "10")
    difficulty_param = request.GET.get("difficulty", "").strip() or None

    try:
        k = max(1, min(50, int(k_param)))
    except ValueError:
        k = 10

    if not username:
        return JsonResponse({
            "status": "error",
            "message": "Username is required. Please provide a valid LeetCode username."
        }, status=400)

    # 1. Fetch user public profile and recent accepted submissions
    raw_profile = fetch_user_profile(username)
    submissions = fetch_submissions(username, limit=50)

    # Extract solved slugs
    solved_slugs = [s["titleSlug"] for s in submissions if s.get("titleSlug")]

    # 2. Compute recommendations using ML pipeline
    recs, target_diff = recommend_for_solved_slugs(
        solved_slugs,
        k=k,
        filter_difficulty=difficulty_param
    )

    # 3. Process User Stats Summary
    user_info = {
        "username": username,
        "exists": raw_profile is not None,
        "realName": raw_profile["profile"].get("realName") if raw_profile and raw_profile.get("profile") else username,
        "avatar": raw_profile["profile"].get("userAvatar") if raw_profile and raw_profile.get("profile") else None,
        "ranking": raw_profile["profile"].get("ranking") if raw_profile and raw_profile.get("profile") else "N/A",
        "reputation": raw_profile["profile"].get("reputation") if raw_profile and raw_profile.get("profile") else 0,
        "recentSubmissionsCount": len(submissions),
        "targetDifficulty": target_diff,
        "solvedBreakdown": {"Easy": 0, "Medium": 0, "Hard": 0}
    }

    if raw_profile and raw_profile.get("submitStats"):
        stats = raw_profile["submitStats"].get("acSubmissionNum", [])
        for item in stats:
            diff = item.get("difficulty")
            if diff in user_info["solvedBreakdown"]:
                user_info["solvedBreakdown"][diff] = item.get("count", 0)

    # Extract top solved topics from recent submissions
    X, problems, mlb, model, slug_map = get_cached_artifacts()
    topic_counter = {}
    for slug in solved_slugs:
        if slug in slug_map:
            for t in slug_map[slug].topics or []:
                topic_counter[t] = topic_counter.get(t, 0) + 1

    sorted_topics = sorted(topic_counter.items(), key=lambda x: x[1], reverse=True)
    user_info["topTopics"] = [t[0] for t in sorted_topics[:4]] if sorted_topics else ["Arrays", "Hashing", "Trees"]

    # 4. Format Recommendations List
    formatted_recs = []
    for rank, res in enumerate(recs, 1):
        p = res["problem"]
        sim_pct = round(res["similarity"] * 100, 1)
        final_pct = round(res["final_score"] * 100, 1)

        # Generate contextual explanation
        top_topic = p.topics[0] if p.topics else "General DSA"
        explanation = f"{final_pct}% match based on your recent activity in {top_topic} and target {target_diff} level."

        formatted_recs.append({
            "rank": rank,
            "id": p.id,
            "leetcode_id": p.leetcode_id,
            "title": p.title,
            "slug": p.slug,
            "difficulty": p.difficulty,
            "acceptance_rate": p.acceptance_rate,
            "topics": p.topics,
            "similarity_pct": sim_pct,
            "final_score_pct": final_pct,
            "explanation": explanation,
            "url": f"https://leetcode.com/problems/{p.slug}/"
        })

    return JsonResponse({
        "status": "success",
        "user": user_info,
        "recommendations": formatted_recs,
        "count": len(formatted_recs)
    })


@require_GET
def similar_problem_api(request):
    """
    API endpoint: GET /api/similar/?problem_id=123&k=6
    Returns content-similar problems using KNN.
    """
    pid_param = request.GET.get("problem_id", "").strip()
    k_param = request.GET.get("k", "6")

    try:
        k = max(1, min(20, int(k_param)))
    except ValueError:
        k = 6

    if not pid_param:
        return JsonResponse({"status": "error", "message": "problem_id is required"}, status=400)

    try:
        if pid_param.isdigit():
            problem = Problem.objects.get(leetcode_id=int(pid_param))
        else:
            problem = Problem.objects.get(slug=pid_param)
    except Problem.DoesNotExist:
        return JsonResponse({"status": "error", "message": f"Problem '{pid_param}' not found"}, status=404)

    similar_results = find_similar(problem.id, k=k)

    formatted = []
    for res in similar_results:
        p = res["problem"]
        sim_pct = round((1.0 - res["distance"]) * 100, 1)
        formatted.append({
            "id": p.id,
            "leetcode_id": p.leetcode_id,
            "title": p.title,
            "slug": p.slug,
            "difficulty": p.difficulty,
            "acceptance_rate": p.acceptance_rate,
            "topics": p.topics,
            "similarity_pct": sim_pct,
            "url": f"https://leetcode.com/problems/{p.slug}/"
        })

    return JsonResponse({
        "status": "success",
        "query_problem": {
            "id": problem.id,
            "leetcode_id": problem.leetcode_id,
            "title": problem.title,
            "slug": problem.slug,
            "difficulty": problem.difficulty,
            "topics": problem.topics
        },
        "similar": formatted
    })


@require_GET
def problem_list_api(request):
    """
    API endpoint: GET /api/problems/?q=binary&difficulty=Medium&limit=30
    Search & filter available problem dataset.
    """
    q = request.GET.get("q", "").strip().lower()
    difficulty = request.GET.get("difficulty", "").strip()
    topic = request.GET.get("topic", "").strip().lower()
    limit = max(1, min(100, int(request.GET.get("limit", "30"))))

    qs = Problem.objects.all()

    if difficulty and difficulty in ["Easy", "Medium", "Hard"]:
        qs = qs.filter(difficulty=difficulty)

    if q:
        qs = qs.filter(title__icontains=q)

    results = []
    for p in qs:
        if topic and not any(topic in t.lower() for t in (p.topics or [])):
            continue
        results.append({
            "id": p.id,
            "leetcode_id": p.leetcode_id,
            "title": p.title,
            "slug": p.slug,
            "difficulty": p.difficulty,
            "acceptance_rate": p.acceptance_rate,
            "topics": p.topics,
            "url": f"https://leetcode.com/problems/{p.slug}/"
        })
        if len(results) >= limit:
            break

    return JsonResponse({
        "status": "success",
        "count": len(results),
        "problems": results
    })


@require_GET
def stats_api(request):
    """
    API endpoint: GET /api/stats/
    Returns overview metrics for dataset.
    """
    total_problems = Problem.objects.count()
    easy_count = Problem.objects.filter(difficulty="Easy").count()
    medium_count = Problem.objects.filter(difficulty="Medium").count()
    hard_count = Problem.objects.filter(difficulty="Hard").count()

    X, problems, mlb, model, slug_map = get_cached_artifacts()
    topics_count = len(mlb.classes_) if mlb is not None else 0

    return JsonResponse({
        "status": "success",
        "total_problems": total_problems,
        "difficulty_breakdown": {
            "Easy": easy_count,
            "Medium": medium_count,
            "Hard": hard_count
        },
        "total_topics": topics_count
    })
