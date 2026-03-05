"""
Build embeddings from SHL assessments for semantic search.
Run this AFTER scraper.py
Usage: python build_embeddings.py
"""

import json
import numpy as np
import pickle
import os
import sys
import io

# Fix Windows console encoding for special characters
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def build_text_for_embedding(assessment):
    """Create rich text representation of each assessment for embedding"""
    parts = []
    
    name = assessment.get("name", "")
    if name:
        parts.append(f"Assessment: {name}")
    
    if assessment.get("description") and "Outdated browser" not in assessment.get("description", ""):
        parts.append(f"Description: {assessment['description']}")
    
    if assessment.get("test_type"):
        type_map = {
            "A": "Ability and Aptitude cognitive test",
            "B": "Biodata and Situational Judgement behavioral test",
            "C": "Competencies evaluation",
            "D": "Development and 360 degree feedback",
            "E": "Assessment Exercises practical test",
            "K": "Knowledge and Skills technical test",
            "P": "Personality and Behavior psychometric test",
            "S": "Simulations hands-on coding test"
        }
        types = [type_map.get(t, t) for t in assessment["test_type"]]
        parts.append(f"Type: {', '.join(types)}")
    
    if assessment.get("duration"):
        parts.append(f"Duration: {assessment['duration']} minutes")
    
    # Add enriched keywords from name for better matching
    name_lower = name.lower()
    
    domain_keywords = []
    if any(w in name_lower for w in ["java", "python", "sql", "c#", "c++", ".net", "angular", "react", "javascript", "selenium"]):
        domain_keywords.extend(["programming", "software development", "coding", "developer", "technical"])
    if any(w in name_lower for w in ["sales", "marketing"]):
        domain_keywords.extend(["selling", "business", "commercial", "revenue"])
    if any(w in name_lower for w in ["manager", "leader", "supervisor", "executive"]):
        domain_keywords.extend(["leadership", "management", "people management"])
    if any(w in name_lower for w in ["entry", "graduate", "apprentice"]):
        domain_keywords.extend(["fresher", "junior", "entry level", "new hire", "campus"])
    if any(w in name_lower for w in ["verify", "numerical", "verbal", "inductive"]):
        domain_keywords.extend(["cognitive", "aptitude", "reasoning", "ability"])
    if any(w in name_lower for w in ["opq", "personality"]):
        domain_keywords.extend(["behavioral", "culture fit", "psychometric"])
    if any(w in name_lower for w in ["automata", "simulation"]):
        domain_keywords.extend(["coding simulation", "hands-on", "practical coding test"])
    if any(w in name_lower for w in ["excel", "word", "microsoft"]):
        domain_keywords.extend(["office", "productivity", "spreadsheet"])
    if any(w in name_lower for w in ["communication", "english", "writing"]):
        domain_keywords.extend(["language", "verbal", "written", "content writer"])
    if any(w in name_lower for w in ["data", "analytics", "tableau", "statistics"]):
        domain_keywords.extend(["data analysis", "analyst", "reporting", "business intelligence"])
    if any(w in name_lower for w in ["admin", "clerk", "assistant"]):
        domain_keywords.extend(["administrative", "clerical", "office support"])
    if any(w in name_lower for w in ["customer", "service"]):
        domain_keywords.extend(["support", "call center", "helpdesk"])
    
    if domain_keywords:
        parts.append(f"Related: {' '.join(domain_keywords)}")
    
    # Name keywords
    name_keywords = name.lower().replace("-", " ").replace("new", "").replace("(", "").replace(")", "")
    parts.append(f"Keywords: {name_keywords}")
    
    return " | ".join(parts)


def build_embeddings_sentence_transformers(assessments):
    """Use sentence-transformers (free, no API key needed)"""
    try:
        from sentence_transformers import SentenceTransformer
        print("Loading sentence-transformers model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and good quality
        
        texts = [build_text_for_embedding(a) for a in assessments]
        print(f"Building embeddings for {len(texts)} assessments...")
        embeddings = model.encode(texts, show_progress_bar=True)
        
        return embeddings, model
    except ImportError:
        print("sentence-transformers not installed. Falling back to TF-IDF.")
        return None, None


def build_tfidf_index(assessments):
    """Fallback: TF-IDF based search (no GPU needed, works offline)"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    texts = [build_text_for_embedding(a) for a in assessments]
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=10000,
        stop_words='english'
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    return vectorizer, tfidf_matrix


if __name__ == "__main__":
    print("=== Building Search Index ===\n")
    
    # Load assessments
    if not os.path.exists("shl_assessments.json"):
        print("ERROR: shl_assessments.json not found. Run scraper.py first!")
        exit(1)
    
    with open("shl_assessments.json") as f:
        assessments = json.load(f)
    
    print(f"Loaded {len(assessments)} assessments")
    
    # Try sentence transformers first
    embeddings, model = build_embeddings_sentence_transformers(assessments)
    
    if embeddings is not None:
        # Save embeddings
        np.save("embeddings.npy", embeddings)
        with open("assessments_index.pkl", "wb") as f:
            pickle.dump({"assessments": assessments, "method": "sentence_transformers"}, f)
        print(f"\n✅ Saved embeddings: embeddings.npy")
        print(f"✅ Saved index: assessments_index.pkl")
    else:
        # Fallback to TF-IDF
        vectorizer, tfidf_matrix = build_tfidf_index(assessments)
        with open("tfidf_index.pkl", "wb") as f:
            pickle.dump({
                "vectorizer": vectorizer,
                "matrix": tfidf_matrix,
                "assessments": assessments,
                "method": "tfidf"
            }, f)
        print(f"\n✅ Saved TF-IDF index: tfidf_index.pkl")
    
    print("\nDone! Ready to run recommender.")
