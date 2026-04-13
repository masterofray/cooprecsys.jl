# HTTP API Reference — HybridRecommender

The `PredictAPI` module exposes a lightweight HTTP/JSON server for real-time inference.

---

## Starting the Server

```bash
# Via bash script (recommended)
./run_predict.sh server --port 8080 --host 0.0.0.0

# Via Julia directly
julia --threads auto --project=. scripts/predict.jl \
  --mode server \
  --svd-model models/svd_model.jls \
  --fp-model  models/fp_model.jls \
  --port 8080
```

---

## Endpoints

### `GET /health`

Liveness probe. Returns server uptime and model readiness.

**Response 200:**
```json
{
  "status": "ok",
  "uptime_s": 3600,
  "svd_ready": true,
  "fp_ready": true,
  "timestamp": "2024-01-15T10:30:00.000"
}
```

---

### `GET /model/info`

Returns metadata about the loaded models.

**Response 200:**
```json
{
  "svd_components": 50,
  "svd_n_users": 9467,
  "svd_n_items": 4659,
  "fp_rules": 12450,
  "fp_min_support": 0.01,
  "fp_min_conf": 0.1,
  "model_version": "1.0.0"
}
```

---

### `POST /predict`

Single-user recommendation with optional cart context.

**Request body:**
```json
{
  "user_id": 12345,
  "cart": [789, 456, 123],
  "top_k_cf": 100,
  "top_k_final": 20
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `user_id` | int/string | ✅ | — | User to recommend for |
| `cart` | array | ❌ | `[]` | Current cart (product IDs) |
| `top_k_cf` | int | ❌ | `100` | Stage 1 candidate count |
| `top_k_final` | int | ❌ | `20` | Final recommendations |

**Response 200:**
```json
{
  "user_id": 12345,
  "cart_size": 3,
  "recommendations": [
    {
      "item_id": 49302,
      "cf_score": 0.923411,
      "fp_lift": 3.14,
      "hybrid_score": 3.820,
      "rank": 1
    },
    {
      "item_id": 13176,
      "cf_score": 0.891200,
      "fp_lift": 0.0,
      "hybrid_score": 0.891200,
      "rank": 2
    }
  ],
  "model_version": "1.0.0",
  "timestamp": "2024-01-15T10:30:05.123"
}
```

**Response 400** — Missing or invalid `user_id`:
```json
{"error": "Missing 'user_id'"}
```

**Response 500** — Internal error:
```json
{"error": "Models not loaded. Call load_models() first."}
```

---

### `POST /batch`

Batch recommendation for multiple users in a single call.

**Request body:**
```json
{
  "requests": [
    {"user_id": 12345, "cart": [789, 456]},
    {"user_id": 67890, "cart": []},
    {"user_id": 11111}
  ]
}
```

**Response 200:**
```json
{
  "count": 3,
  "results": [
    {
      "user_id": 12345,
      "recommendations": [ ... ],
      "timestamp": "2024-01-15T10:30:05.123"
    },
    {
      "user_id": 67890,
      "recommendations": [ ... ],
      "timestamp": "2024-01-15T10:30:05.145"
    },
    {
      "user_id": 11111,
      "error": "Unknown user_id=11111",
      "timestamp": "2024-01-15T10:30:05.160"
    }
  ]
}
```

> ℹ️ Batch errors are per-item — one failed user does not abort the whole batch.

---

## Programmatic Julia API

For embedding in Julia applications without HTTP overhead:

```julia
include("src/HybridRecommender.jl")
using .HybridRecommender.PredictAPI, .HybridRecommender.Reranker

# Step 1: Load models (once at startup)
PredictAPI.load_models("models/svd_model.jls", "models/fp_model.jls")

# Step 2: Configure reranker
cfg = RerankerConfig(
    100,   # top_k_cf    — Stage 1 candidates
    20,    # top_k_final — final recommendations
    256,   # chunk_size  — users per batch
    1.0    # lift_weight — FP lift multiplier
)

# Step 3a: Single prediction
result = PredictAPI.predict_single(12345, [789, 456, 123]; cfg=cfg)
for rec in result["recommendations"]
    println("Item $(rec["item_id"]) — hybrid_score=$(rec["hybrid_score"])")
end

# Step 3b: Batch prediction
requests = [
    Dict("user_id" => 12345, "cart" => [789, 456]),
    Dict("user_id" => 67890, "cart" => Int[])
]
results = PredictAPI.predict_batch(requests; cfg=cfg)
```

---

## Score Interpretation

| Field | Range | Meaning |
|-------|-------|---------|
| `cf_score` | any real | Raw dot-product similarity in latent space. Higher = more similar to user's history. |
| `fp_lift` | ≥ 0.0 | Lift of the best matching FP rule. 0 = no rule fired. >1 = item is associated with cart items. |
| `hybrid_score` | any real | `cf_score × (1 + fp_lift)`. The primary ranking key. |

Items where `fp_lift > 0` have been boosted because they co-occur with something in the user's active cart. Items with `fp_lift = 0` are ranked on CF alone.

---

## Error Codes

| HTTP Status | Cause |
|-------------|-------|
| 200 | Success |
| 400 | Missing required field or invalid JSON |
| 404 | Unknown endpoint |
| 500 | Internal error (model not loaded, computation failure) |

---

## cURL Examples

```bash
# Health
curl -s http://localhost:8080/health | python3 -m json.tool

# Single prediction — user with empty cart
curl -s -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id": 12345}' | python3 -m json.tool

# Single prediction — user with cart context
curl -s -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id": 12345, "cart": [49302, 13176], "top_k_final": 5}' \
  | python3 -m json.tool

# Batch
curl -s -X POST http://localhost:8080/batch \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"user_id": 12345, "cart": [789]},
      {"user_id": 67890, "cart": []}
    ]
  }' | python3 -m json.tool
```
