# Aesthetic Hub — Data Design Document

---

## 1. System Overview

Aesthetic Hub adds a personalized ranking layer to Immich. Aesthetic Hub intercepts search results and re-ranks them per user based on learned aesthetic preferences.

The system learns from user interactions which accumulate as training data for a personalized MLP that is retrained weekly. A global quality model handles new users who have not yet built up enough interaction history.

The data platform supports two loops:

- **Online loop:** A user uploads or searches -> feature-svc computes a CLIP embedding -> serving team runs MLP -> score stored -> Immich re-ranks
- **Offline loop:** Interaction events accumulate -> batch pipeline compiles versioned training datasets -> training team retrains MLP weekly -> new model, user embeddings and scores deployed

---

## 2. Data Repositories

### 2.1 PostgreSQL

Stores live operational state only. Historical bulk data lives in the object store.

Interaction events and inference logs are append-only and only queried in aggregate, so they are written to PostgreSQL and to the bucket. The PostgreSQL tables are pruned and keep only data from the last 30 days, while the bucket retains the full history.

All tables represent current state (latest score per user/asset, current user embeddings, active model version). 

Never update historical events or scores. If a user deletes a photo, we soft delete the event with `deleted_at` but never remove it from the dataset. This preserves the integrity of our training data and audit trail.

#### Tables

- model_versions
- user_embeddings
- user_interaction_counts
- interaction_events
- inference_log
- aesthetic_scores

**`model_versions`** — one row per weekly retrain. A unique partial index on `is_active` enforces exactly one active model at a time.

| Column | Type | Notes |
|---|---|---|
| version_id | VARCHAR PK | e.g. `v2026-04-13` |
| dataset_version | VARCHAR | links to `datasets/v{date}/` in bucket |
| event_cutoff | TIMESTAMPTZ | cutoff used by batch pipeline |
| train_row_count | INTEGER | |
| val_row_count | INTEGER | |
| unique_users | INTEGER | |
| git_sha_training | VARCHAR | training team git SHA |
| git_sha_pipeline | VARCHAR | data team git SHA |
| mlp_object_key | VARCHAR | bucket path to MLP weights |
| embeddings_object_key | VARCHAR | bucket path to user embeddings |
| is_active | BOOLEAN | unique index: only one TRUE at a time |
| activated_at | TIMESTAMPTZ | |
| deactivated_at | TIMESTAMPTZ | null while active |
| created_at | TIMESTAMPTZ | |

**`user_embeddings`** — one row per user. overwritten on every retrain. 64-dimensional learned preference vector.

| Column | Type | Notes |
|---|---|---|
| user_id | VARCHAR PK | |
| embedding | FLOAT4[] | 64-d vector |
| model_version | VARCHAR FK | references model_versions |
| updated_at | TIMESTAMPTZ | |

**`user_interaction_counts`** — fast lookup for cold-start alpha computation. Incremented on every interaction event.

| Column | Type | Notes |
|---|---|---|
| user_id | VARCHAR PK | |
| interaction_count | INT | used to calculate alpha |
| first_seen_at | TIMESTAMPTZ | |
| updated_at | TIMESTAMPTZ | |

**`interaction_events`** — dual-write: PostgreSQL for fast queries + object store parquet for durable bulk storage. Each event is one user interaction with one asset. PostgreSQL keeps a rolling 30-day window; object store keeps permanently.

| Column | Type | Notes |
|---|---|---|
| event_id | VARCHAR PK | deduplication key |
| asset_id | VARCHAR | |
| user_id | VARCHAR | |
| event_type | VARCHAR | favorite / album_add / download / share / view_expanded / archive / delete |
| session_id | VARCHAR | groups events in one browsing session |
| label | FLOAT | 0.0–1.0 training target |
| source | VARCHAR | `holdout_simulation` or `immich_upload` — governance filter |
| model_version | VARCHAR FK | active model when event was logged (nullable — cold start) |
| is_cold_start | BOOLEAN | |
| alpha | FLOAT | exact blending weight at event time |
| generator_run | VARCHAR | UUID identifying the generator run |
| event_time | TIMESTAMPTZ | when event happened |
| ingested_at | TIMESTAMPTZ | when API received it |
| deleted_at | TIMESTAMPTZ | soft delete — NULL means active |

**`inference_log`** — written by feature-svc on every `/score-image` call. PostgreSQL keeps a rolling 30-day window; object store keeps permanently. Links feature computation to aesthetic scores via `request_id` FK.

| Column | Type | Notes |
|---|---|---|
| request_id | VARCHAR PK | serving team uses as FK in aesthetic_scores |
| asset_id | VARCHAR | |
| user_id | VARCHAR | |
| clip_model_version | VARCHAR | `ViT-L/14 openai/clip` |
| model_version | VARCHAR FK | null if cold start |
| is_cold_start | BOOLEAN | |
| alpha | FLOAT | |
| source | VARCHAR | |
| request_received_at | TIMESTAMPTZ | latency measurement start |
| computed_at | TIMESTAMPTZ | latency measurement end |

**`aesthetic_scores`** — written by serving pipeline. Upserted on every inference. Links to `inference_log` via `inference_request_id` FK for full audit trail.

| Column | Type | Notes |
|---|---|---|
| asset_id | VARCHAR | composite PK |
| user_id | VARCHAR | composite PK |
| score | FLOAT | 0.0–1.0 final blended score |
| model_version | VARCHAR FK | |
| is_cold_start | BOOLEAN | |
| alpha | FLOAT | |
| inference_request_id | VARCHAR FK | references inference_log.request_id |
| source | VARCHAR | |
| scored_at | TIMESTAMPTZ | |

### 2.2 Chameleon Object Store — ObjStore_proj21 (CHI@TACC)

Permanent blob storage for all raw data, features, datasets, models, and historical logs. 

Versioned datasets and models are stored under `v{date}/` prefixes. Raw data is stored under `raw-data/` and never modified after ingestion.

```
ObjStore_proj21/
├── raw-data/
│   ├── uhd-iqa/
│   │   ├── images/ 
│   │   ├── uhd-iqa-metadata.csv
│   │   └── manifest.json
│   └── flickr-aes/
│       ├── images/
│       ├── FLICKR-AES_image_score.txt
│       ├── FLICKR-AES_image_labeled_by_each_worker.csv
│       └── manifest.json
│
├── datasets/
│   ├── global-uhd/
|       ├── train.parquet
│       ├── val.parquet
│       ├── test.parquet
│       └── dataset_card.json
│   ├── global-flickr/
|       ├── train.parquet
│       ├── val.parquet
│       ├── test.parquet
│       └── dataset_card.json
│   ├── personalized-flickr/ 
|       ├── train.parquet
│       ├── val.parquet
│       ├── test.parquet
│       ├── new_user_holdout.parquet
│       └── dataset_card.json
│   └── v{date}/                 VERSIONED — batch pipeline output
│       └── personalized-flickr/
│           ├── train.parquet
│           ├── val.parquet
│           ├── test.parquet
│           └── dataset_card.json
│
├── models/
│   ├── clip/
│   │   └── ViT-L-14.pt
│   └── v{date}/                 VERSIONED — training team writes
│       ├── personalized_mlp.onnx
│       ├── user_embeddings.parquet
│       └── model_card.json
│
└── production-sim/
    ├── interactions/
    │   └── date=YYYY-MM-DD/     part-NNNN.parquet — interaction events
    └── inference-log/
        └── date=YYYY-MM-DD/     part-NNNN.parquet — inference audit
```
----

## 3. Versioning Strategy

UHD-IQA raw + parquets, FLICKR-AES raw + parquets, Holdout parquet are never modified after initial upload. They are not versioned.

Retraining datasets: These are stored in obj store at `datasets/v{date}/`. These change weekly on new user interactions.

Models: These are stored in obj store at `models/v{date}/`. Each version corresponds to a weekly retrain. The active model is switched atomically via the `model_versions` table in PostgreSQL.

Interaction events: These are dual-written to PostgreSQL (rolling 30-day window) and to the bucket (permanent). The bucket path is `production-sim/interactions/date=YYYY-MM-DD/part-NNNN.parquet`. Each parquet file contains a batch of events ingested together. 

Inference logs: These are dual-written to PostgreSQL (rolling 30-day window) and to the bucket (permanent). The bucket path is `production-sim/inference-log/date=YYYY-MM-DD/part-NNNN.parquet`. Each parquet file contains a batch of inference logs ingested together.

**Dataset card** — every versioned dataset includes `dataset_card.json`:

```json
{
  "version":            "v2026-04-06",
  "event_cutoff":       "2026-04-06",
  "git_sha":            "a1b2c3d",
  "created_at":         "2026-04-06T20:21:14+00:00",
  "train_rows":         4236,
  "val_rows":           904,
  "test_rows":          919,
  "unique_users":       42,
  "unique_assets":      3847,
  "label_distribution": {"favorite": 312, "view_expanded": 1840, ...},
  "excluded_rows":      {"duplicates_removed": 0, "invalid_label_or_type": 1, ...},
  "content_hash_train": "a3f2...",
  "content_hash_val":   "b7c1...",
  "content_hash_test":  "d9e4..."
}
```

MD5 content hashes per split allow downstream consumers to verify they are using exactly the dataset a model was trained on.

---

## 4. Audit Trail

```
aesthetic_scores.inference_request_id
  → inference_log.request_id
      → inference_log.model_version
          → model_versions.version_id
              → model_versions.dataset_version
                  → datasets/v{date}/personalized-flickr/dataset_card.json
                      → event_cutoff + git_sha + MD5 content hashes
                          → production-sim/interactions/date=*/
                              → individual event_id rows
```

---


## 5. Label Scheme

All labels, model outputs, and stored scores use the 0–1 range.

| Event type | Label | Rationale |
|---|---|---|
| favorite | 1.0 | Strongest positive signal |
| album_add | 0.9 | Very strong positive |
| download | 0.7 | Strong positive |
| share | 0.6 | Moderate positive |
| view_expanded | 0.4 | Weak positive / neutral |
| archive | 0.1 | Weak negative |
| delete | 0.0 | Strongest rejection |
| no interaction | excluded | Not a training example |

---

## 6. Candidate Selection & Leakage Prevention

The batch pipeline applies four filters before splitting:

1. **Time filter** — `event_time < cutoff`. Events on or after the cutoff date are excluded.
2. **Deduplication** — on `event_id`. The API already deduplicates, this is a defensive check.
3. **Eligibility** — valid label ∈ [0,1], valid event_type, non-null user_id and asset_id.
4. **Decontamination** — source ∈ `{holdout_simulation, immich_upload}`, `deleted_at IS NULL`. Excludes soft-deleted events and unknown sources.

**Leakage prevention:**

- **Burst grouping** — events within 60 seconds per user share a `burst_id`. Splits happen at burst level, not event level. This prevents correlated session events from appearing in both train and val.
- **Chronological split** — oldest 70% of bursts per user → train, next 15% → val, newest 15% → test. Time ordering is per user, not global.
- **Quality gate** — asserts `train_max_time < val_min_time` per user. Pipeline exits if violated.

---



## 7. Future Work

| Item | Description |
|---|---|
| Immich poller | Sidecar service polling `GET /api/assets?updatedAfter=` every 60s, calling `/score-image` for new assets |
| Score history archival | Before upserting `aesthetic_scores`, write old row to `production-sim/score-history/date=*/`|