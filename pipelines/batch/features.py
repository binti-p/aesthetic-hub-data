import io
import logging
import os

import boto3
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def get_s3():
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT_URL",
                                    "https://chi.tacc.chameleoncloud.org:7480"),
        aws_access_key_id=os.environ["EC2_ACCESS_KEY"],
        aws_secret_access_key=os.environ["EC2_SECRET_KEY"],
    )


def load_clip_embeddings(container: str) -> pd.DataFrame:
    """
    load clip embeddings from ingestion pipeline parquets.
    these are static — computed once during ingestion, never change.
    """
    s3 = get_s3()
    logger.info("loading CLIP embeddings from object store...")

    dfs = []
    for prefix in [
        "datasets/personalized-flickr/train.parquet",
        "datasets/personalized-flickr/val.parquet",
        "datasets/personalized-flickr/test.parquet",
        "datasets/personalized-flickr/new_user_holdout.parquet",
    ]:
        try:
            buf = io.BytesIO()
            s3.download_fileobj(container, prefix, buf)
            buf.seek(0)
            df = pd.read_parquet(buf)
            if "image_name" in df.columns and "embedding" in df.columns:
                dfs.append(df[["image_name", "embedding"]].rename(
                    columns={"image_name": "asset_id"}
                ))
        except Exception as e:
            logger.warning(f"could not load {prefix}: {e}")

    if not dfs:
        logger.warning("no CLIP embeddings found — clip_embedding will be null")
        return pd.DataFrame(columns=["asset_id", "clip_embedding"])

    embeddings = pd.concat(dfs).drop_duplicates(subset=["asset_id"])
    embeddings = embeddings.rename(columns={"embedding": "clip_embedding"})
    logger.info(f"loaded {len(embeddings)} CLIP embeddings")
    return embeddings


def load_user_embeddings(db_url: str) -> pd.DataFrame:
    """
    load current user embeddings from postgres.
    these are the embeddings active at pipeline run time.
    since we rescore everything on every retrain, we use current embeddings.
    """
    import sqlalchemy as sa
    engine = sa.create_engine(db_url)
    with engine.connect() as conn:
        result = conn.execute(sa.text(
            "SELECT user_id, embedding, model_version FROM user_embeddings"
        ))
        rows = result.fetchall()

    if not rows:
        logger.warning("no user embeddings in postgres — user_embedding will be zeros")
        return pd.DataFrame(columns=["user_id", "user_embedding", "model_version"])

    df = pd.DataFrame(rows, columns=["user_id", "user_embedding", "model_version"])
    logger.info(f"loaded {len(df)} user embeddings from postgres")
    return df


def join_features(
    interactions: pd.DataFrame,
    clip_embeddings: pd.DataFrame,
    user_embeddings: pd.DataFrame,
) -> pd.DataFrame:
    """
    join clip and user embeddings onto interactions.
    users with no embedding get zero vector (cold start).
    """
    USER_EMB_DIM = 64

    # join clip embeddings
    df = interactions.merge(clip_embeddings, on="asset_id", how="left")
    missing_clip = df["clip_embedding"].isna().sum()
    if missing_clip > 0:
        logger.warning(f"{missing_clip} rows missing CLIP embedding — dropping")
        df = df[df["clip_embedding"].notna()]

    # join user embeddings — left join, fill missing with zeros (cold start)
    df = df.merge(
        user_embeddings[["user_id", "user_embedding", "model_version"]],
        on="user_id",
        how="left",
        suffixes=("", "_user")
    )

    # fill cold start users with zero embeddings
    cold_start_mask = df["user_embedding"].isna()
    zero_emb = [0.0] * USER_EMB_DIM
    df["user_embedding"] = df["user_embedding"].astype(object)
    for idx in df[cold_start_mask].index:
        df.at[idx, "user_embedding"] = zero_emb
    df["is_cold_start"] = cold_start_mask

    # use event model_version if user model_version missing
    if "model_version_user" in df.columns:
        df["model_version"] = df["model_version"].fillna(df["model_version_user"]).infer_objects(copy=False)
        df = df.drop(columns=["model_version_user"])

    logger.info(
        f"features joined: {len(df)} rows, "
        f"{cold_start_mask.sum()} cold start users"
    )
    return df
