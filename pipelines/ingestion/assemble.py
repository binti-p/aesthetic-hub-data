import numpy as np
import pandas as pd
import torch
import clip
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

OUTPUT_DIR  = Path("/tmp/aesthetic-hub-output")
CACHE_FILE  = Path("/tmp/clip-embedding-cache.npy")
BATCH_SIZE  = 64
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


def load_clip():
    print(f"loading CLIP ViT-L/14 on {DEVICE}...")
    model, preprocess = clip.load("ViT-L/14", device=DEVICE)
    model.eval()
    return model, preprocess


def load_cache():
    if CACHE_FILE.exists():
        cache = np.load(CACHE_FILE, allow_pickle=True).item()
        print(f"resuming from cache: {len(cache)} embeddings already done")
        return cache
    return {}


def save_cache(cache):
    np.save(CACHE_FILE, cache)


@torch.no_grad()
def compute_embeddings(image_paths, model, preprocess, cache):
    missing = [p for p in image_paths if str(p) not in cache]
    print(f"{len(image_paths)} images total, {len(missing)} need embedding")

    saved_at = 0
    for start in tqdm(range(0, len(missing), BATCH_SIZE), desc="embedding"):
        batch = missing[start:start + BATCH_SIZE]
        tensors, valid = [], []
        for p in batch:
            try:
                tensors.append(preprocess(Image.open(p).convert("RGB")))
                valid.append(p)
            except Exception:
                pass

        if not tensors:
            continue

        feats = model.encode_image(torch.stack(tensors).to(DEVICE))
        feats = (feats / feats.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)

        for path, emb in zip(valid, feats):
            cache[str(path)] = emb

        if len(cache) - saved_at >= 500:
            save_cache(cache)
            saved_at = len(cache)

    save_cache(cache)
    return cache


def write_parquet(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = []
    for col in df.columns:
        if col == "embedding":
            schema.append(pa.field(col, pa.list_(pa.float32())))
        elif col in ("score",):
            schema.append(pa.field(col, pa.float32()))
        else:
            schema.append(pa.field(col, pa.string()))
    pq.write_table(pa.Table.from_pandas(df, schema=pa.schema(schema)), str(path), compression="snappy")
    print(f"  {len(df):,} rows -> {path.relative_to(OUTPUT_DIR)}")


def add_embeddings(df, cache, path_col="image_path"):
    df["embedding"] = df[path_col].apply(lambda p: cache.get(str(p)))
    before = len(df)
    df = df.dropna(subset=["embedding"]).reset_index(drop=True)
    if len(df) < before:
        print(f"  dropped {before - len(df)} rows with missing embeddings")
    return df


def assemble_global_uhd(uhd_df, cache):
    print("\nassembling global-uhd...")
    df = add_embeddings(uhd_df, cache)
    for split in ["train", "val", "test"]:
        write_parquet(
            df[df["split"] == split][["image_name", "embedding", "score", "split"]],
            OUTPUT_DIR / "global-uhd" / f"{split}.parquet"
        )


def assemble_global_flickr(scores_df, cache):
    print("\nassembling global-flickr...")
    df = add_embeddings(scores_df, cache)
    for split in ["train", "val", "test"]:
        write_parquet(
            df[df["split"] == split][["image_name", "embedding", "global_score", "split"]].rename(columns={"global_score": "score"}),
            OUTPUT_DIR / "global-flickr" / f"{split}.parquet"
        )


def assemble_personalized_flickr(workers_df, cache):
    print("\nassembling personalized-flickr...")
    df = add_embeddings(workers_df, cache)

    for split in ["train", "val", "test"]:
        write_parquet(
            df[df["split"] == split][["image_name", "worker_id", "embedding", "worker_score_norm", "split"]]
              .rename(columns={"worker_id": "user_id", "worker_score_norm": "score"}),
            OUTPUT_DIR / "personalized-flickr" / f"{split}.parquet"
        )

    holdout = df[df["split"] == "production_new_user"][
        ["worker_id", "image_name", "image_path", "embedding", "worker_score_norm"]
    ].rename(columns={"worker_id": "user_id", "worker_score_norm": "score", "image_path": "s3_url"})
    holdout["s3_url"] = holdout["s3_url"].astype(str)

    write_parquet(
        holdout[["user_id", "image_name", "s3_url", "embedding", "score"]],
        OUTPUT_DIR / "personalized-flickr" / "new_user_holdout.parquet"
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from normalize import load_uhd, load_flickr

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    uhd_df = load_uhd()
    scores_df, workers_df = load_flickr()

    model, preprocess = load_clip()
    cache = load_cache()

    all_paths = list(set(
        list(uhd_df["image_path"].astype(str)) +
        list(scores_df["image_path"].astype(str))
    ))

    cache = compute_embeddings([Path(p) for p in all_paths], model, preprocess, cache)

    assemble_global_uhd(uhd_df, cache)
    assemble_global_flickr(scores_df, cache)
    assemble_personalized_flickr(workers_df, cache)

    print("\ndone. next: run upload.py")
