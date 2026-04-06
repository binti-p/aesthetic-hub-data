import hashlib
import io
import json
import logging
import os
from datetime import datetime, timezone

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "event_id", "asset_id", "user_id",
    "clip_embedding", "user_embedding",
    "label", "event_type", "event_time",
    "model_version", "is_cold_start", "alpha",
    "source", "split", "burst_id"
]


def get_s3():
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT_URL",
                                    "https://chi.tacc.chameleoncloud.org:7480"),
        aws_access_key_id=os.environ["EC2_ACCESS_KEY"],
        aws_secret_access_key=os.environ["EC2_SECRET_KEY"],
    )


def _df_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    table = pa.Table.from_pandas(df)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    buf.seek(0)
    return buf.read()


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def write_datasets(
    df: pd.DataFrame,
    container: str,
    version: str,
    cutoff: str,
    git_sha: str,
    candidate_stats: dict,
):
    s3 = get_s3()
    prefix = f"datasets/{version}/personalized-flickr"

    # keep only output columns that exist
    cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    df = df[cols]

    splits = {}
    hashes = {}

    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split].drop(columns=["split"])
        data = _df_to_parquet_bytes(split_df)
        key  = f"{prefix}/{split}.parquet"
        s3.put_object(Bucket=container, Key=key, Body=data)
        splits[split] = len(split_df)
        hashes[f"content_hash_{split}"] = _md5(data)
        logger.info(f"wrote {len(split_df)} rows → {key}")

    # dataset card
    label_dist = df["event_type"].value_counts().to_dict()

    card = {
        "version":          version,
        "event_cutoff":     cutoff,
        "git_sha":          git_sha,
        "created_at":       datetime.now(timezone.utc).isoformat(),
        "train_rows":       splits["train"],
        "val_rows":         splits["val"],
        "test_rows":        splits["test"],
        "unique_users":     int(df["user_id"].nunique()),
        "unique_assets":    int(df["asset_id"].nunique()),
        "label_distribution": label_dist,
        "excluded_rows":    candidate_stats,
        **hashes,
    }

    card_key = f"{prefix}/dataset_card.json"
    s3.put_object(
        Bucket=container,
        Key=card_key,
        Body=json.dumps(card, indent=2).encode()
    )
    logger.info(f"wrote dataset card → {card_key}")
    logger.info(json.dumps(card, indent=2))

    return card
