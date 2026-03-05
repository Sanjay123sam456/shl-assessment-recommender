## SHL Assessment Recommendation System – Approach (2‑Page Summary)

### 1. Problem Understanding & Goals

The task is to build an SHL assessment recommendation system that, given:
- a **natural language query**,
- a **job description (JD) text**, or
- a **URL containing a JD**,

returns **5–10 relevant “Individual Test Solutions”** (not pre‑packaged job solutions). The solution is evaluated using **Mean Recall@10** on a hidden test set, plus qualitative review of the pipeline, LLM usage, and engineering quality.

My goals:
- Robust, reproducible **data pipeline** (scraping → cleaning → embeddings).
- **Retrieval + LLM reranking** for relevance.
- Clean **API + frontend** that matches the spec.
- Clear **evaluation & iteration story** around Mean Recall@10.

---

### 2. Data Pipeline

**Catalogue Scraping (`scraper.py`)**

- Scrapes the SHL catalogue from  
  `https://www.shl.com/solutions/products/product-catalog/` using `?start=<offset>&type=1`.
- `type=1` filters to **Individual Test Solutions**, which automatically excludes **Pre‑packaged Job Solutions**, matching the instructions.
- Paginates in steps of 12 items, with a safety limit and early stop on consecutive empty pages.
- For each assessment row it extracts:
  - `name`, canonical SHL `url`.
  - `test_type` codes (A/B/C/D/E/K/P/S).
  - `remote_support` and `adaptive_support` from icon/“Yes” patterns.

If fewer than **377** assessments are scraped, I merge in a fixed list of **known URLs** from the training data to guarantee at least 377 Individual Test Solutions. The final `shl_assessments.json` used in this submission contains **398** assessments.

**Detail Enrichment**

- Optionally visits each assessment page to extract:
  - A short **description** from content/description containers.
  - **Duration** in minutes (regex on page text).
  - Missing `test_type` codes from tag/badge elements.
- Descriptions are truncated to 500 characters to keep embeddings compact.

**Embedding Index (`build_embeddings.py`)**

- Builds a rich text string per assessment via `build_text_for_embedding`:
  - `Assessment: <name>`
  - `Description: …` (if available and meaningful)
  - Natural‑language expansion of each `test_type`
  - Extra domain keywords inferred from the name (e.g., Java/Python/SQL → programming; Verify → cognitive; OPQ → personality).
- Encodes these texts with **`sentence-transformers/all-MiniLM-L6-v2`**.
- Saves:
  - `embeddings.npy` (embedding matrix).
  - `assessments_index.pkl` (`{"assessments": [...], "method": "sentence_transformers"}`).

This gives a compact, reusable semantic index over the SHL catalogue.

---

### 3. Recommendation Engine

**Base Retrieval (`recommender.py`)**

- `SHLRecommender` loads `embeddings.npy` + `assessments_index.pkl` and the same `all-MiniLM-L6-v2` model.
- `_hybrid_search(query, top_k=30)`:
  - Encodes the query.
  - Computes cosine similarity to all assessments.
  - Sorts by similarity and returns the top 30 candidates.

This already implements a modern, embedding‑based retrieval layer over the scraped catalogue.

**LLM Reranking with Gemini**

- `_call_gemini_reranker` (optional, gated by `GEMINI_API_KEY`):
  - Formats up to 15 top candidates with name, human‑readable test types, and duration.
  - Asks **Gemini 2.0 Flash** to “Select the N most relevant assessments” and return a JSON array of indices.
  - Parses indices, clamps to available candidates, and caches results to stay efficient.
- If the key is missing or the call fails, it falls back to identity ranking (no rerank).

This step lets the model refine the ranking based on richer semantics than cosine similarity alone.

**Type Balance for Mixed Queries**

The brief requires that, when queries span multiple domains (e.g., technical + behavioral), recommendations should be **balanced** (e.g., both K and P types).

- `_needs_balance(query)`:
  - Detects **technical intent** via words like *developer, engineer, Java, Python, SQL, knowledge, skills*.
  - Detects **personality/behavior intent** via words like *personality, culture, collaborate, teamwork, stakeholders, communication*.
- `_enforce_type_balance(query, candidates, final, n)`:
  - If the query signals both technical and behavioral needs:
    - Checks whether the current `final` list includes at least one **K** and one **P** test.
    - For any missing type, it scans remaining high‑scoring candidates, pulls in the best matching assessment of that type, and trims back to `n`.
- `recommend(query, n=10)`:
  - Runs semantic search → Gemini rerank → greedy fill → type balance.

This makes queries like “Java developer who can collaborate with business teams” more likely to return a mix of **Knowledge & Skills (K)** and **Personality & Behavior (P)** tests, as requested.

---

### 4. API & URL Handling

**Endpoints (`api.py`)**

- `GET /health` → `{"status": "healthy"}`.
- `POST /recommend`:
  - Request: `{"query": "<natural language, JD text, or JD URL>"}`.
  - Response: `{"recommended_assessments": [...]}` with fields:
    - `name`, `url`, `test_type`, `duration`, `remote_support`, `adaptive_support`, `description`.

**URL → JD Text**

- If `query` starts with `http://` or `https://`:
  - Fetches the page with `requests`.
  - Parses HTML with `BeautifulSoup("lxml")`.
  - Extracts paragraph text (`<p>`), filters for longer sentences, and concatenates into JD text.
  - Passes this extracted JD text into the recommender.
- If fetching fails, returns a clear 400 error.

This satisfies the requirement that the API can accept **URLs containing job descriptions**.

---

### 5. Frontend

Files: `frontend.html`, `app.py`

- `app.py` runs FastAPI (Uvicorn) in a background thread and serves the HTML via Streamlit, injecting `window.API_BASE` for the JS to call.
- `frontend.html`:
  - SHL‑styled UI with tabs for **Job Description / Natural Language / URL**.
  - Textarea input and a **“Get Recommendations”** button that calls `/recommend`.
  - Renders metrics (number of assessments, types, average duration, remote‑ready count) and cards with SHL links.
  - Provides a **“Download JSON”** action to export the current recommendations.

The same HTML can be used in deployment; only `API_BASE` must point to the hosted API.

---

### 6. Evaluation & Predictions

**Metric & Implementation (`evaluate.py`)**

- Uses the provided **`Gen_AI_Dataset.xlsx`**:
  - `Train-Set` (columns: `Query`, `Assessment_url`).
  - `Test-Set` (column: `Query`).
- For each unique train query:
  - Calls `/recommend`.
  - Computes **Recall@10** by comparing URL slugs (`.../view/<slug>/`) to handle `/products/` vs `/solutions/products/`.
- Aggregates into **Mean Recall@10**.
- Current result on the train set: **Mean Recall@10 ≈ 0.2556** (baseline that can be improved further).

**Submission CSV**

- For the test set, the script generates `predictions.csv` in exactly the format from Appendix 3:
  - Header: `Query,Assessment_url`
  - One row per (query, recommended URL), with queries repeated for multiple recommendations.

This file is ready to be scored by SHL’s automated pipeline.

---

### 7. Deployment & Submission Checklist

- **API deployment**: run `uvicorn api:app --host 0.0.0.0 --port $PORT` on a cloud platform, with `GEMINI_API_KEY` set and the index files present.
- **Frontend deployment**: run `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` and set `API_BASE` to the deployed API URL.

Submission items the assignment asks for:
- Public **API URL** (with `/health` and `/recommend` functioning).
- Public **frontend URL**.
- **GitHub repo** URL with full code and experiments.
- **`predictions.csv`** in the prescribed format.
- This **2‑page approach document** summarizing the architecture and optimization efforts.

