## Dataset
 
- **Source:** [Taobao User Behavior Dataset](https://tianchi.aliyun.com/dataset/649) (Alibaba / Tianchi) — used in published academic research.
- **Scale:** ~100,150,807 interaction records (`User_ID`, `Product_ID`, `Category_ID`, `Behavior`, `Timestamp`).
- **Behaviors:** `pv` (page view), `cart`, `fav` (favorite), `buy`.
### The sparsity problem
 
An initial random sample of 500K rows produced an average of only **1.4 interactions per user** — far too sparse for a recommendation model to learn meaningful patterns, with a severe class imbalance (`pv` 89% vs `buy` 2%).
 
### Solution: Smart Certified Sample
 
Instead of sampling random rows, the pipeline filters for **active users with 15+ total interactions** in the full dataset (939,456 eligible users), then randomly samples 20,000 of them.
 
| Metric | Random Sample | Smart Certified Sample |
|---|---|---|
| Users | ~297,000 | 20,000 |
| Avg. interactions/user | 1.4 | **105.6** |
| Buy interactions | 8,711 | **42,003** |
 
**Certification check:** behavior-rate distribution of the sample vs. the full 100M-row dataset differs by less than **0.02%** across `pv`/`fav`/`buy` rates, with identical Top-10 category rankings — confirming the sample is statistically representative, not biased toward a subset of behavior.
 
## Data Cleaning
 
| Step | Action | Reason |
|---|---|---|
| 1 | Remove `cart` behavior | No equivalent concept in a donation platform |
| 2 | Convert `Timestamp` → `DateTime` | Enables time-based filtering and evaluation |
| 3 | Remove invalid dates (outside Nov 25 – Dec 3, 2017) | Matches the dataset's actual collection window |
| 4 | Add interaction score (`pv=1`, `fav=8`, `buy=20`) | Weights real donations far above passive views |
| 5 | Remove duplicate rows | Data quality |
| 6 | Filter products with <5 interactions | Reduces noise; 590K → 82K products, increases matrix density 3.5x |
 
## Model
 
**Architecture:** Hybrid Recommendation System
 
| Component | Weight | Method |
|---|---|---|
| Collaborative Filtering | 60% | `TruncatedSVD` (scikit-learn), `n_components=500` |
| Content-Based Filtering | 40% | User-Category interaction profile |
 
```
Hybrid Score = (0.6 × SVD cosine similarity) + (0.4 × Category preference score)
```
 
**Why TruncatedSVD?** `scikit-surprise` and `implicit` (ALS) were both evaluated first but failed to build on Python 3.13 due to Cython/CMake compilation issues with no available C compiler. `TruncatedSVD` works natively with `scipy.sparse` matrices and required no additional native dependencies — a more practical choice given the constraint, while still being a well-established matrix factorization technique.
 
**Sparse matrix:** the User-Item matrix (19,906 × 82,023) is built using `scipy.sparse.csr_matrix`, with a density of only 0.04% — storing the matrix densely would require ~13GB of RAM versus ~7MB with the sparse representation.
 
## Evaluation
 
- **Split strategy:** Time-based (train = interactions before Dec 1, 2017; test = interactions after) — chosen over a random split to simulate realistic future-prediction conditions rather than allowing information leakage from later interactions into training.
- **Metric level:** Category-level rather than product-level, since with 82,023 unique products, exact product-level matches are statistically near-impossible to achieve meaningfully.
| K | Precision@K | Recall@K | F1@K |
|---|---|---|---|
| 5 | 12.78% | 47.55% | 0.2015 |
| 10 | 7.48% | 55.75% | 0.1319 |
| 20 | 4.17% | 61.97% | 0.0782 |
 
### Baseline comparison
 
To verify the model learns genuine user preferences (rather than simply recommending whatever is popular), it was benchmarked against a Popularity baseline:
 
| Metric | Popularity Baseline | Hybrid Model | Improvement |
|---|---|---|---|
| F1@5 | 0.0357 | 0.2015 | **+464% (≈4.6x)** |
 
## API
 
Built with **Flask** + **Flask-CORS**, served with **Gunicorn**, deployed on **Render**.
 
### Endpoints
 
| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Health check |
| `/recommend?user_id={id}` | `GET` | Returns top-5 recommended items for a user |
 
### Example response
 
```json
{
  "user_id": 915552,
  "recommendations": [
    { "product_id": 1381239, "category_id": 4145813, "score": 0.9728 },
    { "product_id": 2842934, "category_id": 4145813, "score": 0.9524 }
  ]
}
```
 
### Deployment footprint
 
The full trained model (`hybrid_model_final.pkl`) was originally 708MB — too large for free-tier hosting. Dimensionality was reduced and serialization split into separate files to bring the total deployment size down to ~127MB:
 
| File | Size | Content |
|---|---|---|
| `encoders.pkl` | 7MB | LabelEncoders for users, products, categories |
| `user_factors.npy` | 15MB | SVD user latent vectors |
| `item_factors.npy` | 62MB | SVD item latent vectors |
| `smart_filtered.csv` | 43MB | Filtered training data, used for category lookups |
 
## Frontend Integration
 
Since Firebase Auth UIDs are strings and the model expects numeric IDs from its training data, a deterministic hash function maps any Firebase UID to a stable number:
 
```js
function uidToNumericId(uid) {
  let hash = 5381;
  for (let i = 0; i < uid.length; i++) {
    hash = ((hash << 5) + hash) ^ uid.charCodeAt(i);
  }
  return Math.abs(hash) % 9_999_999;
}
```
 
A three-tier fallback strategy handles all user states on the frontend:
 
| Tier | Condition | Behavior |
|---|---|---|
| 1 — Returning donor | Has donation history in Firestore | Filter real campaigns by donated categories/institutions, reranked by API confidence score |
| 2 — New user (cold-start) | No history, API returns scores | Show popular active campaigns, reranked by API confidence score |
| 3 — No data | No history, API unavailable | Hide the recommendation section entirely |
 
Reranking blends model confidence with real platform signal:
 
```
Final Score = (API confidence score × 0.5) + (donor popularity × 0.5)
```
 
## Tech Stack
 
`Python` · `scikit-learn` · `scipy` · `pandas` · `numpy` · `Flask` · `Gunicorn` · `Render` · `Firebase` · `React`
 
## Project Context
 
This API powers the recommendation feature of a larger graduation project: a donation platform connecting university students in need (tuition, laptops, food, medicine) with donors, built with React, Flutter, and Firebase. A separate bilingual AI chatbot (DeepSeek API) consumes this API's output to provide recommendation-aware conversational assistance.




