"""
SHL Assessment Recommender Engine
Hybrid: Semantic Search + Keyword Boost + Gemini LLM Reranking
"""

import json
import numpy as np
import pickle
import os
import re
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations"
}

PRIMARY_BOOSTS = {
    "java": ["java", "core-java"],
    "python": ["python"],
    "sql": ["sql", "automata-sql"],
    "javascript": ["javascript"],
    ".net": [".net", "asp", "wcf", "wpf", "xaml"],
    "selenium": ["selenium"],
    "html": ["htmlcss", "html"],
    "css": ["css", "htmlcss"],
    "data": ["data", "sql", "analytics"],
    "analyst": ["data", "sql", "analytics"],
    "tableau": ["tableau"],
    "excel": ["excel", "microsoft-excel"],
    "sales": ["sales", "entry-level-sales", "sales-representative", "technical-sales"],
    "marketing": ["marketing", "digital-advertising", "seo"],
    "seo": ["seo", "search-engine-optimization"],
    "content": ["content", "writing", "writex", "written"],
    "leader": ["leadership", "enterprise-leadership"],
    "manager": ["manager", "leadership", "enterprise-leadership"],
    "executive": ["leadership", "enterprise-leadership"],
    "english": ["english", "verbal", "svar", "comprehension"],
    "communication": ["communication", "interpersonal", "english", "verbal"],
    "writing": ["writing", "writex", "written-english"],
    "cognitive": ["verify", "numerical", "verbal", "inductive", "deductive", "reasoning"],
    "aptitude": ["verify", "numerical", "verbal", "inductive", "reasoning"],
    "numerical": ["numerical", "verify-numerical", "calculation"],
    "personality": ["personality", "opq", "occupational-personality"],
    "culture": ["personality", "opq"],
    "admin": ["admin", "administrative", "clerk", "data-entry"],
    "bank": ["bank", "financial", "cashier"],
    "graduate": ["entry-level", "apprentice", "graduate"],
    "entry": ["entry-level", "apprentice"],
    "customer": ["customer", "service", "support"],
    "testing": ["testing", "manual-testing", "selenium"],
    "coding": ["coding", "programming"],
    "developer": ["programming"],
    "consultant": ["professional", "verify", "opq"],
    "cobol": ["cobol"],
}

SECONDARY_BOOSTS = {
    "sql": ["data", "database"],
    "javascript": ["front-end", "htmlcss"],
    "selenium": ["testing", "manual-testing"],
    "data": ["python", "tableau", "excel", "statistics"],
    "analyst": ["python", "tableau", "excel"],
    "sales": ["business-communication"],
    "marketing": ["writex", "content"],
    "leader": ["opq", "global-skills"],
    "manager": ["opq", "global-skills", "jfa"],
    "executive": ["opq", "global-skills"],
    "english": ["communication", "written"],
    "communication": ["svar"],
    "customer": ["communication", "english"],
    "coding": ["automata"],
    "developer": ["automata"],
    "consultant": ["administrative", "verbal", "numerical"],
}

_gemini_cache = {}
_CACHE_MAX = 200
BASE_DIR = Path(__file__).resolve().parent


def _cache_key(query: str, candidate_names: List[str]) -> str:
    return query.strip().lower()[:200] + "|" + ",".join(candidate_names[:15])


def _call_gemini_reranker(query: str, candidates: List[Dict], n: int = 10) -> List[int]:

    if os.getenv("DISABLE_GEMINI", "0") == "1":
        return list(range(min(n, len(candidates))))

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return list(range(min(n, len(candidates))))

    candidate_names = [c.get("name", "") for c in candidates]

    key = _cache_key(query, candidate_names)

    if key in _gemini_cache:
        return _gemini_cache[key]

    import google.generativeai as genai

    genai.configure(api_key=api_key)

    max_candidates = min(15, len(candidates))

    candidate_lines = []

    for i in range(max_candidates):

        c = candidates[i]

        types = [TYPE_MAP.get(t, t) for t in c.get("test_type", [])]

        dur = c.get("duration", "unknown")

        candidate_lines.append(
            f"{i}. {c['name']} (Types: {', '.join(types)}, Duration: {dur})"
        )

    prompt = f"""
Select the {n} most relevant SHL assessments.

QUERY:
{query}

CANDIDATES:
{chr(10).join(candidate_lines)}

Return JSON array of indices.
"""

    try:

        model = genai.GenerativeModel("gemini-2.0-flash")

        response = model.generate_content(prompt, request_options={"timeout": 8})

        text = response.text.strip()

        match = re.search(r'\[[\d\s,]+\]', text)

        if match:

            indices = json.loads(match.group())

            result = [i for i in indices if i < max_candidates][:n]

            if len(_gemini_cache) >= _CACHE_MAX:
                _gemini_cache.pop(next(iter(_gemini_cache)))

            _gemini_cache[key] = result

            return result

    except Exception:
        pass

    return list(range(min(n, len(candidates))))


class SHLRecommender:

    def __init__(self):

        self.assessments = []

        self.embeddings = None

        self.model = None
        self.vectorizer = None
        self.doc_matrix = None

        self._load_index()

    def _load_index(self):
        embeddings_path = BASE_DIR / "embeddings.npy"
        index_path = BASE_DIR / "assessments_index.pkl"
        json_path = BASE_DIR / "shl_assessments.json"

        if embeddings_path.exists() and index_path.exists():

            self.embeddings = np.load(str(embeddings_path))

            with open(index_path, "rb") as f:

                data = pickle.load(f)

            self.assessments = data["assessments"]

            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
            except Exception:
                self.model = None
                self._init_tfidf_fallback()

        elif json_path.exists():

            # Deployment-safe fallback: build an in-memory TF-IDF index directly
            # from scraped assessment data when precomputed embedding files are absent.
            with open(json_path, "r", encoding="utf-8") as f:
                self.assessments = json.load(f)

            self.model = None
            self._init_tfidf_fallback()

        else:

            raise FileNotFoundError("Missing index files and shl_assessments.json. Run scraper.py first.")

    def _assessment_text(self, a: Dict) -> str:
        return " ".join(
            [
                a.get("name", ""),
                a.get("description", ""),
                " ".join(TYPE_MAP.get(t, t) for t in a.get("test_type", [])),
            ]
        )

    def _init_tfidf_fallback(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = [self._assessment_text(a) for a in self.assessments]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=40000)
        self.doc_matrix = self.vectorizer.fit_transform(texts)

    def _hybrid_search(self, query: str, top_k: int = 30):

        from sklearn.metrics.pairwise import cosine_similarity

        if self.model is not None:
            query_emb = self.model.encode([query])
            scores = cosine_similarity(query_emb, self.embeddings)[0]
        else:
            if self.vectorizer is None or self.doc_matrix is None:
                self._init_tfidf_fallback()
            query_vec = self.vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self.doc_matrix)[0]

        results = []

        for i, score in enumerate(scores):

            a = self.assessments[i]

            results.append({**a, "score": score})

        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    def _needs_balance(self, query: str) -> Dict[str, bool]:
        """Heuristic: does the query ask for technical (K) and/or personality/behavior (P)?"""
        q = query.lower()
        technical_keywords = [
            "developer", "engineer", "programmer", "coding", "java", "python", "sql",
            "c++", "c#", "javascript", "technical", "knowledge", "skills", "technology"
        ]
        personality_keywords = [
            "personality", "behavior", "behaviour", "culture", "values",
            "collaborate", "collaboration", "teamwork", "stakeholder",
            "communication", "soft skill", "soft skills", "opq"
        ]
        wants_k = any(k in q for k in technical_keywords)
        wants_p = any(k in q for k in personality_keywords)
        return {"K": wants_k, "P": wants_p}

    def _enforce_type_balance(self, query: str, candidates: List[Dict], final: List[Dict], n: int) -> List[Dict]:
        """
        Ensure that for mixed queries (technical + personality),
        the final list contains at least one K and one P assessment when possible.
        """
        needs = self._needs_balance(query)
        if not (needs["K"] and needs["P"]):
            return final[:n]

        has_k = any("K" in a.get("test_type", []) for a in final)
        has_p = any("P" in a.get("test_type", []) for a in final)

        missing_types = []
        if not has_k:
            missing_types.append("K")
        if not has_p:
            missing_types.append("P")

        if not missing_types:
            return final[:n]

        # Search remaining candidates (high to low score) for missing types
        used_urls = {a.get("url") for a in final}
        extras: List[Dict] = []
        for t in missing_types:
            for c in candidates:
                if c.get("url") in used_urls:
                    continue
                if t in c.get("test_type", []):
                    extras.append(self._format_assessment(c))
                    used_urls.add(c.get("url"))
                    break

        balanced = final + extras
        return balanced[:n]

    def recommend(self, query: str, n: int = 10):

        candidates = self._hybrid_search(query, top_k=30)
        
        reranked = _call_gemini_reranker(query, candidates, n=n)

        final: List[Dict] = []

        seen = set()

        for idx in reranked:

            if idx < len(candidates) and idx not in seen:

                final.append(self._format_assessment(candidates[idx]))

                seen.add(idx)

        for i, c in enumerate(candidates):

            if len(final) >= n:

                break

            if i not in seen:

                final.append(self._format_assessment(c))

        # Enforce K+P balance for mixed queries when possible
        final = self._enforce_type_balance(query, candidates, final, n)

        return final[:n]

    def _format_assessment(self, assessment: Dict) -> Dict:

        duration = assessment.get("duration")

        if not isinstance(duration, int):
            duration = None

        return {
            "name": assessment.get("name", "Unknown"),
            "url": assessment.get("url", ""),
            "adaptive_support": assessment.get("adaptive_support", "No"),
            "description": assessment.get("description", ""),
            "duration": duration,
            "remote_support": assessment.get("remote_support", "Yes"),
            "test_type": assessment.get("test_type", [])
        }


_recommender = None


def get_recommender():

    global _recommender

    if _recommender is None:

        _recommender = SHLRecommender()

    return _recommender


if __name__ == "__main__":

    rec = SHLRecommender()

    queries = [
        "Java developer",
        "Python SQL analyst",
        "Cognitive and personality tests"
    ]

    for q in queries:

        print("\nQuery:", q)

        results = rec.recommend(q)

        for r in results:

            print("-", r["name"], "|", r["url"])
