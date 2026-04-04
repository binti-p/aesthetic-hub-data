import numpy as np
import pandas as pd
from pathlib import Path

SEED                  = 42
GLOBAL_RATIOS         = {"train": 0.70, "val": 0.10, "test": 0.10, "production": 0.10}
SEEN_IMAGE_RATIOS     = {"train": 0.70, "val": 0.10, "test": 0.10, "production_seen": 0.10}
NEW_USER_WORKER_RATIO = 0.20

UHD_DIR    = Path("/data/uhd-iqa")
FLICKR_DIR = Path("/data/flickr-aes")


def split_items(items, ratios, seed=42):
    items = np.array(sorted(list(items)))
    rng   = np.random.default_rng(seed)
    items = items[rng.permutation(len(items))]

    names  = list(ratios.keys())
    vals   = np.array(list(ratios.values()), dtype=float)
    vals  /= vals.sum()
    counts = np.floor(vals * len(items)).astype(int)
    counts[-1] = len(items) - counts[:-1].sum()

    splits, start = {}, 0
    for name, count in zip(names, counts):
        splits[name] = items[start:start + count]
        start += count
    return splits


def load_scores_txt(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2 and parts[0].lower().endswith(".jpg"):
                try:
                    rows.append((parts[0], float(parts[1])))
                except ValueError:
                    continue
    df = pd.DataFrame(rows, columns=["image_name", "global_score"])
    return df.drop_duplicates(subset=["image_name"]).reset_index(drop=True)


def load_uhd():
    print("loading uhd-iqa...")
    df = pd.read_csv(UHD_DIR / "uhd-iqa-metadata.csv")
    df["split"] = df["set"].map({"training": "train", "validation": "val", "test": "test"})
    folder_map  = {"train": "training", "val": "validation", "test": "test"}
    df["image_path"] = df.apply(lambda r: UHD_DIR / folder_map[r["split"]] / r["image_name"], axis=1)
    df = df[df["image_path"].apply(Path.exists)].reset_index(drop=True)
    print(f"  {len(df)} images, splits: {df['split'].value_counts().to_dict()}")
    return df[["image_name", "image_path", "quality_mos", "split"]].rename(columns={"quality_mos": "score"})


def load_flickr():
    print("loading flickr-aes...")
    scores_df  = load_scores_txt(FLICKR_DIR / "FLICKR-AES_image_score.txt")
    workers_df = pd.read_csv(FLICKR_DIR / "FLICKR-AES_image_labeled_by_each_worker.csv", skipinitialspace=True)
    workers_df.columns = [c.strip() for c in workers_df.columns]
    workers_df = workers_df.rename(columns={"imagePair": "image_name", "worker": "worker_id", "score": "worker_score"})
    workers_df["worker_score_norm"] = (workers_df["worker_score"].astype(float) - 1.0) / 4.0

    image_lookup = {p.name: p for p in (FLICKR_DIR / "40K").glob("*.jpg")}
    print(f"  {len(image_lookup)} images on disk")

    scores_df["image_path"]  = scores_df["image_name"].map(image_lookup)
    workers_df["image_path"] = workers_df["image_name"].map(image_lookup)
    scores_df  = scores_df[scores_df["image_path"].notna()].reset_index(drop=True)
    workers_df = workers_df[workers_df["image_path"].notna()].reset_index(drop=True)

    # global splits
    global_split_map = {img: s for s, imgs in split_items(scores_df["image_name"].unique(), GLOBAL_RATIOS, SEED).items() for img in imgs}
    scores_df["split"] = scores_df["image_name"].map(global_split_map)

    # personalized splits
    all_workers = np.array(sorted(workers_df["worker_id"].unique()))
    rng         = np.random.default_rng(SEED)
    perm        = rng.permutation(len(all_workers))
    n_holdout   = max(1, int(round(NEW_USER_WORKER_RATIO * len(all_workers))))
    new_user_workers = set(all_workers[perm[:n_holdout]])
    seen_workers     = set(all_workers[perm[n_holdout:]])

    seen_df = workers_df[workers_df["worker_id"].isin(seen_workers)]
    seen_split_map = {img: s for s, imgs in split_items(seen_df["image_name"].unique(), SEEN_IMAGE_RATIOS, SEED).items() for img in imgs}

    mask = workers_df["worker_id"].isin(seen_workers)
    workers_df["split"] = None
    workers_df.loc[mask,  "split"] = workers_df.loc[mask,  "image_name"].map(seen_split_map)
    workers_df.loc[~mask, "split"] = "production_new_user"

    print(f"  seen workers: {len(seen_workers)}, holdout: {len(new_user_workers)}")
    print(f"  splits: {workers_df['split'].value_counts().to_dict()}")
    return scores_df, workers_df


if __name__ == "__main__":
    uhd_df = load_uhd()
    scores_df, workers_df = load_flickr()
    print("done")
