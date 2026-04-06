import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

BURST_GAP_SECONDS = 60
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15


def assign_bursts(df: pd.DataFrame) -> pd.DataFrame:
    """
    group events within 60 seconds per user into the same burst.
    prevents leakage from correlated session events being split
    across train/val/test.
    """
    df = df.sort_values(["user_id", "event_time"]).copy()
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)

    burst_ids = []
    burst_counter = 0

    for user_id, group in df.groupby("user_id"):
        times = group["event_time"].values
        burst_id = burst_counter

        for i, t in enumerate(times):
            if i == 0:
                burst_ids.append(f"{user_id}_{burst_id:04d}")
            else:
                gap = (pd.Timestamp(t) - pd.Timestamp(times[i-1])).total_seconds()
                if gap > BURST_GAP_SECONDS:
                    burst_counter += 1
                    burst_id = burst_counter
                burst_ids.append(f"{user_id}_{burst_id:04d}")
            
        burst_counter += 1

    df["burst_id"] = burst_ids
    logger.info(f"assigned {df['burst_id'].nunique()} bursts to {df['user_id'].nunique()} users")
    return df


def chronological_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    chronological split per user — oldest events to train, newest to test.
    split at burst level to keep correlated events together.
    ensures model never sees future interactions during training.
    """
    df = df.copy()
    df["split"] = "train"

    for user_id, group in df.groupby("user_id"):
        bursts = (
            group.groupby("burst_id")["event_time"]
            .min()
            .sort_values()
            .index.tolist()
        )
        n = len(bursts)
        train_end = int(n * TRAIN_FRAC)
        val_end   = int(n * (TRAIN_FRAC + VAL_FRAC))

        train_bursts = set(bursts[:train_end])
        val_bursts   = set(bursts[train_end:val_end])
        test_bursts  = set(bursts[val_end:])

        df.loc[
            (df["user_id"] == user_id) & (df["burst_id"].isin(val_bursts)),
            "split"
        ] = "val"
        df.loc[
            (df["user_id"] == user_id) & (df["burst_id"].isin(test_bursts)),
            "split"
        ] = "test"

    counts = df["split"].value_counts()
    logger.info(f"split: train={counts.get('train',0)} val={counts.get('val',0)} test={counts.get('test',0)}")

    # quality gate — lecture requirement
    for user_id, group in df.groupby("user_id"):
        train = group[group["split"] == "train"]["event_time"]
        val   = group[group["split"] == "val"]["event_time"]
        if len(train) > 0 and len(val) > 0:
            assert train.max() < val.min(), \
                f"leakage detected for user {user_id}: train max > val min"

    logger.info("quality gate passed: no temporal leakage detected")
    return df
