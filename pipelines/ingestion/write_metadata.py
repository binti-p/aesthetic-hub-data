"""
write_metadata.py — writes manifest.json and dataset_card.json files
to ObjStore_proj21 using rclone.

run after assemble.py and upload.py are done.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

now = datetime.now(timezone.utc).isoformat()
OUT = Path("/tmp/aesthetic-hub-metadata")
OUT.mkdir(exist_ok=True)


def write(data, path):
    full = OUT / path
    full.parent.mkdir(parents=True, exist_ok=True)
    with open(full, "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {path}")


# manifests
write({
    "dataset":     "UHD-IQA",
    "source":      "https://datasets.vqa.mmsp-kn.de/archives/UHD-IQA/UHD-IQA-database.zip",
    "produced_at": now,
    "produced_by": "ingestion pipeline",
    "contents": [
        "raw-data/uhd-iqa/images/ (6073 images)",
        "raw-data/uhd-iqa/uhd-iqa-metadata.csv"
    ]
}, "raw-data/uhd-iqa/manifest.json")

write({
    "dataset":     "FLICKR-AES",
    "source":      "https://drive.google.com/drive/folders/1LR6trJhN4XbgTtqZo1zfe272cAkXqA7e",
    "produced_at": now,
    "produced_by": "ingestion pipeline",
    "contents": [
        "raw-data/flickr-aes/images/ (~40,984 images)",
        "raw-data/flickr-aes/FLICKR-AES_image_score.txt",
        "raw-data/flickr-aes/FLICKR-AES_image_labeled_by_each_worker.csv"
    ]
}, "raw-data/flickr-aes/manifest.json")

# dataset cards
base = {
    "produced_at":      now,
    "produced_by":      "ingestion pipeline",
    "clip_model":       "ViT-L/14",
    "clip_library":     "openai/clip",
    "split_basis":      "reproduced from training team colab (seed=42)",
    "leakage_controls": [
        "holdout users (20% of annotators) excluded from train/val/test",
        "splits reproduced with fixed seed=42 matching training team colab",
    ],
    "note": "initial dataset — static, not versioned"
}

write({**base,
    "dataset_variant":   "global-uhd",
    "label_description": "quality_mos (0-1) from uhd-iqa-metadata.csv",
    "split_counts":      {"train": 4269, "val": 904, "test": 900},
}, "datasets/global-uhd/dataset_card.json")

write({**base,
    "dataset_variant":   "global-flickr",
    "label_description": "averaged annotator score (0-1) from FLICKR-AES_image_score.txt",
    "split_counts":      {"train": 28349, "val": 4049, "test": 4049, "production": 4052},
}, "datasets/global-flickr/dataset_card.json")

write({**base,
    "dataset_variant":   "personalized-flickr",
    "label_description": "per-worker score normalized (worker_score - 1) / 4",
    "split_counts":      {"train": 109019, "val": 15596, "test": 15453,
                          "production_seen": 15526, "production_new_user": 37614},
    "holdout_users":     42,
    "seen_users":        168,
}, "datasets/personalized-flickr/dataset_card.json")

print("all metadata written to", OUT)
