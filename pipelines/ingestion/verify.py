import sys
from pathlib import Path
import pandas as pd


def check(condition, ok_msg, fail_msg):
    if condition:
        print(f"  [OK] {ok_msg}")
        return True
    print(f"  [FAIL] {fail_msg}")
    return False


def verify_uhd():
    print("\n=== UHD-IQA ===")
    base = Path("/data/uhd-iqa")
    ok = True

    meta = pd.read_csv(base / "uhd-iqa-metadata.csv")
    ok &= check(len(meta) == 6073, f"row count: {len(meta)}", f"unexpected row count: {len(meta)}")
    ok &= check(
        set(meta["set"].unique()) == {"training", "validation", "test"},
        f"split values ok",
        f"unexpected splits: {meta['set'].unique()}"
    )
    ok &= check(meta["quality_mos"].between(0, 1).all(), "scores in [0, 1]", "scores out of range")

    for split, expected in [("training", 4269), ("validation", 904), ("test", 900)]:
        count = len(list((base / split).glob("*")))
        ok &= check(count == expected, f"{split}/: {count} files", f"{split}/: expected {expected}, got {count}")

    return ok


def verify_flickr():
    print("\n=== FLICKR-AES ===")
    base = Path("/data/flickr-aes")
    ok = True

    rows = []
    with open(base / "FLICKR-AES_image_score.txt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                rows.append(float(parts[1]))
    ok &= check(len(rows) > 40000, f"score rows: {len(rows)}", f"too few score rows: {len(rows)}")

    workers = pd.read_csv(base / "FLICKR-AES_image_labeled_by_each_worker.csv", skipinitialspace=True)
    workers.columns = [c.strip() for c in workers.columns]
    ok &= check(len(workers) > 200000, f"worker rows: {len(workers)}", f"too few worker rows: {len(workers)}")
    ok &= check(workers["score"].between(1, 5).all(), "worker scores in [1, 5]", "worker scores out of range")
    print(f"  [OK] unique annotators: {workers['worker'].nunique()}")

    count = len(list((base / "40K").glob("*.jpg")))
    ok &= check(count > 40000, f"image count: {count}", f"too few images: {count}")

    return ok


if __name__ == "__main__":
    uhd_ok = verify_uhd()
    flickr_ok = verify_flickr()
    print("\n" + "=" * 50)
    if uhd_ok and flickr_ok:
        print("all checks passed")
    else:
        print("some checks failed")
        sys.exit(1)
