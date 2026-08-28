from django.urls import path
from recommender import views

urlpatterns = [
    path("recommend/", views.recommend_user_api, name="api_recommend"),
    path("similar/", views.similar_problem_api, name="api_similar"),
    path("problems/", views.problem_list_api, name="api_problems"),
    path("stats/", views.stats_api, name="api_stats"),
]
