#!/usr/bin/env python3
"""
batch pipeline — compiles versioned training and evaluation datasets
from production interaction events.

run:
  python pipeline.py --cutoff 2026-04-06 --version v2026-04-06

candidate selection:
  - time range: events before cutoff only
  - deduplication: on event_id
  - eligibility: valid label, event_type, non-null ids
  - decontamination: source filter, soft delete respect

leakage prevention:
  - 60-second burst grouping per user
  - chronological split per user (70/15/15)
  - quality gate: assert train_max < val_min per user
  - user embeddings from postgres (current, since full rescore on retrain)
"""
import argparse
import io
import logging
import os
import subprocess
import sys

import boto3
import pandas as pd
from dotenv import load_dotenv

from candidate import select_candidates
from features import load_clip_embeddings, load_user_embeddings, join_features
from splits import assign_bursts, chronological_split
from writer import write_datasets

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def get_s3():
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT_URL",
                                    "https://chi.tacc.chameleoncloud.org:7480"),
        aws_access_key_id=os.environ["EC2_ACCESS_KEY"],
        aws_secret_access_key=os.environ["EC2_SECRET_KEY"],
    )


def load_interactions(container: str, cutoff: str) -> pd.DataFrame:
    """load all interaction parquets from object store up to cutoff date."""
    s3 = get_s3()
    prefix = "production-sim/interactions/"

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=container, Prefix=prefix)

    dfs = []
    files_loaded = 0

    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".parquet"):
                continue

            # date partition filter — skip obviously future partitions
            # format: production-sim/interactions/date=YYYY-MM-DD/part-NNNN.parquet
            try:
                date_part = key.split("date=")[1].split("/")[0]
                if date_part >= cutoff[:10]:
                    continue
            except (IndexError, ValueError):
                pass

            buf = io.BytesIO()
            s3.download_fileobj(container, key, buf)
            buf.seek(0)
            dfs.append(pd.read_parquet(buf))
            files_loaded += 1

    if not dfs:
        logger.error("no interaction parquets found")
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"loaded {files_loaded} parquet files: {len(df):,} rows")
    return df


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd="/app"
        ).decode().strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Aesthetic Hub batch pipeline")
    parser.add_argument("--cutoff",    required=True,
                        help="event cutoff date e.g. 2026-04-06")
    parser.add_argument("--version",   required=True,
                        help="dataset version e.g. v2026-04-06")
    parser.add_argument("--container", default=os.getenv("OBJSTORE_CONTAINER",
                                                          "ObjStore_proj21"))
    args = parser.parse_args()

    db_url = os.environ["DATABASE_URL"]

    logger.info("=" * 70)
    logger.info("AESTHETIC HUB BATCH PIPELINE")
    logger.info(f"cutoff:    {args.cutoff}")
    logger.info(f"version:   {args.version}")
    logger.info(f"container: {args.container}")
    logger.info("=" * 70)

    # step 1 — load interactions
    logger.info("step 1: loading interactions...")
    interactions = load_interactions(args.container, args.cutoff)

    # step 2 — candidate selection
    logger.info("step 2: candidate selection...")
    interactions, candidate_stats = select_candidates(interactions, args.cutoff)

    if len(interactions) == 0:
        logger.error("no candidates after selection — exiting")
        sys.exit(1)

    # step 3 — join features
    logger.info("step 3: joining features...")
    clip_embeddings = load_clip_embeddings(args.container)
    user_embeddings = load_user_embeddings(db_url)
    df = join_features(interactions, clip_embeddings, user_embeddings)

    # step 4 — burst grouping
    logger.info("step 4: burst grouping...")
    df = assign_bursts(df)

    # step 5 — chronological split
    logger.info("step 5: chronological split...")
    df = chronological_split(df)

    # step 6 — write output
    logger.info("step 6: writing output...")
    git_sha = get_git_sha()
    card = write_datasets(
        df=df,
        container=args.container,
        version=args.version,
        cutoff=args.cutoff,
        git_sha=git_sha,
        candidate_stats=candidate_stats,
    )

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"train: {card['train_rows']} val: {card['val_rows']} test: {card['test_rows']}")
    logger.info(f"output: datasets/{args.version}/personalized-flickr/")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
