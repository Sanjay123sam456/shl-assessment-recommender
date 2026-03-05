"""Quick test for the /recommend API endpoint"""
import requests
import json

url = "http://localhost:8000/recommend"

queries = [
    "I am hiring for Java developers who can also collaborate effectively with my business teams.",
    "Looking to hire mid-level professionals who are proficient in Python, SQL and JavaScript.",
    "I need cognitive and personality tests for an analyst role"
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    r = requests.post(url, json={"query": query}, timeout=60)
    print(f"Status: {r.status_code}")
    
    if r.status_code == 200:
        data = r.json()
        assessments = data.get("recommended_assessments", [])
        print(f"Recommendations: {len(assessments)}")
        for i, a in enumerate(assessments):
            print(f"  {i+1}. {a['name']}")
            print(f"     Types: {a['test_type']} | Duration: {a.get('duration', 'N/A')}")
            print(f"     URL: {a['url']}")
    else:
        print(f"Error: {r.text}")
