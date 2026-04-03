"""
download.py — downloads raw datasets to the VM

FLICKR-AES comes from Google Drive (gdown)
UHD-IQA comes from a direct download link
"""
import subprocess
import zipfile
import urllib.request
from pathlib import Path

FLICKR_DIR = Path("/data/flickr-aes")
UHD_DIR    = Path("/data/uhd-iqa")

FLICKR_FOLDER_ID = "1LR6trJhN4XbgTtqZo1zfe272cAkXqA7e"
UHD_ZIP_URL = "https://datasets.vqa.mmsp-kn.de/archives/UHD-IQA/UHD-IQA-database.zip"


def run(cmd):
    print(f"running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {cmd}")


def download_flickr():
    print("--- FLICKR-AES ---")
    FLICKR_DIR.mkdir(parents=True, exist_ok=True)

    print("downloading from Google Drive...")
    run(f"~/.local/bin/gdown --folder {FLICKR_FOLDER_ID} -O {FLICKR_DIR}/")

    # gdown sometimes creates a subfolder — if so, move files up and remove it
    subdirs = [d for d in FLICKR_DIR.iterdir() if d.is_dir() and d.name != "40K"]
    if subdirs:
        subdir = subdirs[0]
        print(f"flattening subfolder: {subdir.name}")
        for f in subdir.iterdir():
            f.rename(FLICKR_DIR / f.name)
        subdir.rmdir()

    # unzip the images
    zips = list(FLICKR_DIR.glob("*.zip"))
    if not zips:
        raise FileNotFoundError("no zip file found after download")

    print(f"unzipping {zips[0].name}, this will take a few minutes...")
    with zipfile.ZipFile(zips[0], "r") as z:
        z.extractall(FLICKR_DIR)
    zips[0].unlink()

    print("done. contents:")
    run(f"ls {FLICKR_DIR}")


def download_uhd():
    print("--- UHD-IQA ---")
    UHD_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = UHD_DIR / "UHD-IQA-database.zip"

    print(f"downloading from {UHD_ZIP_URL}")
    print("this is a large file, will take a while...")

    def progress(block_num, block_size, total_size):
        mb_done  = min(block_num * block_size, total_size) / 1024 / 1024
        mb_total = total_size / 1024 / 1024
        print(f"\r  {mb_done:.0f} / {mb_total:.0f} MB", end="", flush=True)

    urllib.request.urlretrieve(UHD_ZIP_URL, zip_path, reporthook=progress)
    print()

    print(f"unzipping {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(UHD_DIR)
    zip_path.unlink()

    print("done. contents:")
    run(f"ls {UHD_DIR}")


if __name__ == "__main__":
    download_flickr()
    download_uhd()
    print("all downloads complete")