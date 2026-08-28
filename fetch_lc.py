import json
import time
import requests

URL = "https://leetcode.com/graphql"

QUERY = """
query problemsetQuestionList(
    $categorySlug: String,
    $limit: Int,
    $skip: Int,
    $filters: QuestionListFilterInput
) {
    problemsetQuestionList: questionList(
        categorySlug: $categorySlug
        limit: $limit
        skip: $skip
        filters: $filters
    ) {
        total: totalNum
        questions: data {
            acRate
            difficulty
            frontendQuestionId: questionFrontendId
            title
            titleSlug
            topicTags {
                name
                slug
            }
        }
    }
}
"""

HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0"
}


def fetch_problems():
    all_problems = []
    skip = 0
    limit = 100

    while True:
        variables = {
            "categorySlug": "",
            "skip": skip,
            "limit": limit,
            "filters": {}
        }

        response = requests.post(
            URL,
            json={
                "query": QUERY,
                "variables": variables
            },
            headers=HEADERS,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        result = data["data"]["problemsetQuestionList"]
        problems = result["questions"]
        total = result["total"]

        all_problems.extend(problems)

        print(f"Fetched {len(all_problems)}/{total}")

        if len(all_problems) >= total or not problems:
            break

        skip += limit
        time.sleep(0.5)

    return all_problems


if __name__ == "__main__":
    problems = fetch_problems()

    with open("problems.json", "w", encoding="utf-8") as f:
        json.dump(problems, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(problems)} problems.")