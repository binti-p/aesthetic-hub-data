import ssl
import subprocess
import zipfile
import urllib.request
from pathlib import Path

FLICKR_DIR       = Path("/data/flickr-aes")
UHD_DIR          = Path("/data/uhd-iqa")
FLICKR_FOLDER_ID = "1LR6trJhN4XbgTtqZo1zfe272cAkXqA7e"
UHD_ZIP_URL      = "https://datasets.vqa.mmsp-kn.de/archives/UHD-IQA/UHD-IQA-database.zip"


def run(cmd):
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {cmd}")


def flatten_dir(directory):
    for subdir in [d for d in directory.iterdir() if d.is_dir()]:
        for f in subdir.iterdir():
            f.rename(directory / f.name.strip())
        subdir.rmdir()


def download_flickr():
    if (FLICKR_DIR / "40K").exists():
        print("flickr already downloaded, skipping")
        return

    print("downloading flickr-aes...")
    FLICKR_DIR.mkdir(parents=True, exist_ok=True)
    run(f"~/.local/bin/gdown --folder {FLICKR_FOLDER_ID} -O {FLICKR_DIR}/")
    flatten_dir(FLICKR_DIR)

    zips = [f for f in FLICKR_DIR.glob("*.zip") if "FLICKR" in f.name]
    if not zips:
        raise FileNotFoundError(f"no flickr zip found in {FLICKR_DIR}")

    print(f"unzipping {zips[0].name}...")
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(FLICKR_DIR)
    zips[0].unlink()

    for f in FLICKR_DIR.glob("*.zip"):
        f.unlink()

    print(f"done: {len(list((FLICKR_DIR / '40K').glob('*.jpg')))} images")


def download_uhd():
    if (UHD_DIR / "uhd-iqa-metadata.csv").exists():
        print("uhd-iqa already downloaded, skipping")
        return

    print("downloading uhd-iqa...")
    UHD_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = UHD_DIR / "UHD-IQA-database.zip"

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))

    def progress(block_num, block_size, total_size):
        done = min(block_num * block_size, total_size) / 1024 / 1024
        total = total_size / 1024 / 1024
        print(f"\r  {done:.0f} / {total:.0f} MB", end="", flush=True)

    with opener.open(UHD_ZIP_URL) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        block_num = 0
        with open(zip_path, "wb") as f:
            while chunk := response.read(8192):
                f.write(chunk)
                block_num += 1
                progress(block_num, 8192, total_size)
    print()

    print("unzipping...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(UHD_DIR)
    zip_path.unlink()

    for item in (UHD_DIR / "UHD-IQA-database").iterdir():
        item.rename(UHD_DIR / item.name)
    (UHD_DIR / "UHD-IQA-database").rmdir()
    if (UHD_DIR / "__MACOSX").exists():
        import shutil
        shutil.rmtree(UHD_DIR / "__MACOSX")

    print("done")


if __name__ == "__main__":
    download_flickr()
    download_uhd()
    print("all downloads complete")
