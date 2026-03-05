"""
Evaluation Script
- Computes Mean Recall@10 on train set
- Generates predictions CSV for test set submission

Run: python evaluate.py
"""

import pandas as pd
import json
import requests
import os
import sys
import io
from dotenv import load_dotenv

load_dotenv()

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from recommender import get_recommender

API_URL = os.getenv("API_URL", "http://localhost:8000")

def get_url_slug(url: str) -> str:
    """Extract the assessment slug from URL for comparison"""
    return url.rstrip("/").split("/")[-1].lower()

def recall_at_k(predicted_urls: list, relevant_urls: list, k: int = 10) -> float:
    """Compute Recall@K for a single query"""
    if not relevant_urls:
        return 0.0
    predicted_k = predicted_urls[:k]
    
    # Normalize URLs by extracting slug (handles /products/ vs /solutions/products/ mismatch)
    pred_normalized = set(get_url_slug(u) for u in predicted_k)
    rel_normalized  = set(get_url_slug(u) for u in relevant_urls)
    
    hits = len(pred_normalized & rel_normalized)
    return hits / len(rel_normalized)


def get_recommendations_for_query(query: str) -> list:
    """Call the API and get recommendation URLs"""
    try:
        resp = requests.post(
            f"{API_URL}/recommend",
            json={"query": query},
            timeout=60
        )
        if resp.status_code == 200:
            data = resp.json()
            return [a["url"] for a in data.get("recommended_assessments", [])]
    except Exception as e:
        print(f"  API error: {e}")
        
        # Fallback to direct recommender
        rec = get_recommender()
        results = rec.recommend(query, n=10)
        return [r["url"] for r in results]
    
    return []


def evaluate_on_train(excel_path: str = "Gen_AI_Dataset.xlsx"):
    """Compute Mean Recall@10 on the labeled train set"""
    print("=== Evaluating on Train Set ===\n")
    
    df = pd.read_excel(excel_path, sheet_name="Train-Set")
    
    # Group by query
    grouped = df.groupby("Query")["Assessment_url"].apply(list).reset_index()
    
    recalls = []
    for _, row in grouped.iterrows():
        query = row["Query"]
        relevant_urls = row["Assessment_url"]
        
        print(f"Query: {query[:80]}...")
        predicted = get_recommendations_for_query(query)
        
        recall = recall_at_k(predicted, relevant_urls, k=10)
        recalls.append(recall)
        
        print(f"  Relevant: {len(relevant_urls)} | Predicted: {len(predicted)} | Recall@10: {recall:.3f}")
        print()
    
    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    print(f"\n{'='*50}")
    print(f"Mean Recall@10: {mean_recall:.4f}")
    print(f"{'='*50}")
    
    return mean_recall


def generate_test_predictions(excel_path: str = "Gen_AI_Dataset.xlsx",
                               output_path: str = "predictions.csv"):
    """Generate CSV predictions for the unlabeled test set"""
    print("\n=== Generating Test Set Predictions ===\n")
    
    df = pd.read_excel(excel_path, sheet_name="Test-Set")
    test_queries = df["Query"].tolist()
    
    rows = []
    for i, query in enumerate(test_queries):
        print(f"[{i+1}/{len(test_queries)}] {query[:80]}...")
        
        predicted_urls = get_recommendations_for_query(query)
        
        # Each query gets multiple rows (one per recommendation)
        for url in predicted_urls:
            rows.append({
                "Query": query,
                "Assessment_url": url
            })
        
        print(f"  Generated {len(predicted_urls)} recommendations")
    
    predictions_df = pd.DataFrame(rows)
    predictions_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved predictions to {output_path}")
    print(f"   Total rows: {len(predictions_df)}")
    print(f"   Queries: {len(test_queries)}")
    
    return predictions_df


if __name__ == "__main__":
    import sys
    
    # Copy dataset to working dir if needed
    dataset_path = "Gen_AI_Dataset.xlsx"
    if not os.path.exists(dataset_path):
        print(f"ERROR: {dataset_path} not found. Copy the dataset here.")
        sys.exit(1)
    
    # Step 1: Evaluate on train set
    mean_recall = evaluate_on_train(dataset_path)
    
    # Step 2: Generate test predictions
    predictions = generate_test_predictions(dataset_path, "predictions.csv")
    
    print("\n=== DONE ===")
    print(f"Mean Recall@10 on train: {mean_recall:.4f}")
    print("predictions.csv ready for submission")
