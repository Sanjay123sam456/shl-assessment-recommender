# SHL Assessment Recommendation System
### Built for SHL Research Engineer Take-Home Assessment

---

## SETUP — Do This First (15 minutes)

### Step 1: Create project folder and add your API key

```bash
mkdir shl_recommender
cd shl_recommender
```

Create a `.env` file:
```
OPENROUTER_API_KEY=your_openrouter_key_here
```

Get your free OpenRouter key at: https://openrouter.ai/

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

This takes 5-10 minutes first time (downloads sentence-transformers model ~90MB)

### Step 3: Copy dataset

Copy `Gen_AI_Dataset.xlsx` into the `shl_recommender` folder.

---

## BUILD — Run in This Exact Order

### Step 1: Scrape SHL catalogue (10-15 minutes)

```bash
python scraper.py
```

This scrapes 377+ Individual Test Solutions from SHL website.
Output: `shl_assessments.json`

**Verify:** Check that shl_assessments.json has 377+ entries:
```bash
python -c "import json; data=json.load(open('shl_assessments.json')); print(len(data), 'assessments')"
```

### Step 2: Build search index (3-5 minutes)

```bash
python build_embeddings.py
```

Output: `embeddings.npy` and `assessments_index.pkl`

### Step 3: Start API server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Test it: http://localhost:8000/health  
Should return: `{"status": "healthy"}`

### Step 4: Start frontend (new terminal)

```bash
streamlit run app.py
```

Opens at: http://localhost:8501

### Step 5: Run evaluation + generate predictions

```bash
python evaluate.py
```

Output:
- Prints Mean Recall@10 on train set
- Saves `predictions.csv` for submission

---

## DEPLOY TO RENDER (Free)

### Deploy API:
1. Push code to GitHub
2. Go to render.com → New Web Service
3. Connect your repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
6. Add env variable: `OPENROUTER_API_KEY=your_key`

### Deploy Frontend:
1. Same repo, new Web Service on Render
2. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

---

## SUBMISSION CHECKLIST

- [ ] API endpoint: `https://your-api.onrender.com/recommend`
- [ ] Frontend URL: `https://your-frontend.onrender.com`
- [ ] GitHub repo URL (public)
- [ ] `predictions.csv` (query + Assessment_url columns)
- [ ] 2-page approach document

### Test API before submitting:
```bash
curl -X POST "https://your-api.onrender.com/recommend" \
  -H "Content-Type: application/json" \
  -d '{"query": "I am hiring Java developers who collaborate with business teams"}'
```

---

## ARCHITECTURE OVERVIEW

```
User Query / JD Text / URL
        ↓
[URL Fetcher] if URL given
        ↓
[Type Detector] — detects if cognitive/personality/technical needed
        ↓
[Semantic Search] — sentence-transformers cosine similarity
        ↓ top 30 candidates
[LLM Reranker] — OpenRouter Mistral reranks to top 10
        ↓
[Balance Enforcer] — ensures type balance (K+P if both needed)
        ↓
Final 5-10 Recommendations
```

---

## FILES

| File | Purpose |
|------|---------|
| `scraper.py` | Scrapes SHL catalogue → shl_assessments.json |
| `build_embeddings.py` | Builds vector index from assessments |
| `recommender.py` | Core recommendation logic |
| `api.py` | FastAPI backend (/health + /recommend) |
| `app.py` | Streamlit frontend |
| `evaluate.py` | Computes Recall@10, generates predictions.csv |
| `requirements.txt` | Python dependencies |

---

## KNOWN ISSUES & SOLUTIONS

**Issue:** scraper gets fewer than 377 assessments
**Fix:** SHL may paginate differently. Add `?start=X&type=1` parameter manually and check network tab in browser.

**Issue:** OpenRouter rate limit
**Fix:** Use `google/gemma-2-9b-it:free` as alternative free model in recommender.py

**Issue:** Render deployment slow cold start
**Fix:** Set `RENDER_KEEP_ALIVE=true` or use a cron ping service
