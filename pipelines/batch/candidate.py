import logging
import pandas as pd

logger = logging.getLogger(__name__)

VALID_EVENT_TYPES = {
    "favorite", "album_add", "download",
    "share", "view_expanded", "archive", "delete"
}
VALID_SOURCES = {"holdout_simulation", "immich_upload"}


def select_candidates(df: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    original = len(df)
    stats = {}

    # 1. time range — only events before cutoff
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, format="ISO8601")
    cutoff_ts = pd.Timestamp(cutoff, tz="UTC")
    df = df[df["event_time"] < cutoff_ts]
    stats["after_cutoff_filter"] = len(df)

    # 2. deduplication on event_id — defensive, API already deduplicates
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["event_id"], keep="first")
    stats["duplicates_removed"] = before_dedup - len(df)

    # 3. eligibility — valid labels, event types, non-null ids
    before_eligibility = len(df)
    df = df[
        (df["label"] >= 0.0) & (df["label"] <= 1.0) &
        (df["event_type"].isin(VALID_EVENT_TYPES)) &
        (df["user_id"].notna()) &
        (df["asset_id"].notna())
    ]
    stats["invalid_label_or_type"] = before_eligibility - len(df)

    # 4. decontamination — source filter + soft delete
    before_decon = len(df)
    df = df[df["source"].isin(VALID_SOURCES)]
    if "deleted_at" in df.columns:
        df = df[df["deleted_at"].isna()]
    stats["decontaminated"] = before_decon - len(df)

    logger.info(f"candidate selection: {original} → {len(df)} rows")
    logger.info(f"  cutoff filter:      kept {stats['after_cutoff_filter']}")
    logger.info(f"  duplicates removed: {stats['duplicates_removed']}")
    logger.info(f"  invalid removed:    {stats['invalid_label_or_type']}")
    logger.info(f"  decontaminated:     {stats['decontaminated']}")

    return df, stats
