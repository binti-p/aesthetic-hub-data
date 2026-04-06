#!/usr/bin/env bash
# run_ingestion.sh — full ingestion pipeline for Aesthetic Hub
# Run on a fresh Chameleon m1.xlarge VM after cloning the repo.
# Only manual prerequisites:
#   export EC2_ACCESS_KEY=...
#   export EC2_SECRET_KEY=...

set -e

CONTAINER="${OBJSTORE_CONTAINER:-ObjStore_proj21}"

echo "============================================================"
echo "AESTHETIC HUB INGESTION PIPELINE"
echo "container: ${CONTAINER}"
echo "============================================================"

# validate credentials are set
if [ -z "$EC2_ACCESS_KEY" ] || [ -z "$EC2_SECRET_KEY" ]; then
    echo "ERROR: EC2_ACCESS_KEY and EC2_SECRET_KEY must be set"
    exit 1
fi

echo ""
echo "=== Setup: directories ==="
sudo mkdir -p /data/uhd-iqa /data/flickr-aes
sudo chown -R cc /data

echo ""
echo "=== Setup: Python dependencies ==="
pip install pandas numpy pyarrow tqdm pillow boto3 python-dotenv --break-system-packages
pip install git+https://github.com/openai/CLIP.git --break-system-packages
pip install torch torchvision --break-system-packages
pip install gdown --break-system-packages

echo ""
echo "=== Setup: rclone ==="
curl https://rclone.org/install.sh | sudo bash

echo ""
echo "=== Setup: rclone config ==="
mkdir -p ~/.config/rclone
cat > ~/.config/rclone/rclone.conf << EOF
[chi_tacc]
type = s3
provider = Ceph
access_key_id = ${EC2_ACCESS_KEY}
secret_access_key = ${EC2_SECRET_KEY}
endpoint = https://chi.tacc.chameleoncloud.org:7480
no_check_bucket = true
EOF
rclone lsd chi_tacc:

echo ""
echo "=== Step 1: Download datasets ==="
python3 download.py

echo ""
echo "=== Step 2: Verify downloads ==="
python3 verify.py

echo ""
echo "=== Step 3: Compute CLIP embeddings + assemble parquets ==="
python3 assemble.py

echo ""
echo "=== Step 4: Upload CLIP weights to object store ==="
CLIP_CACHE="${HOME}/.cache/clip/ViT-L-14.pt"
if [ ! -f "$CLIP_CACHE" ]; then
    echo "ERROR: CLIP weights not found at $CLIP_CACHE"
    echo "Did assemble.py complete successfully?"
    exit 1
fi
rclone copy "$CLIP_CACHE" chi_tacc:${CONTAINER}/models/clip/ --progress
echo "CLIP weights uploaded to ${CONTAINER}/models/clip/ViT-L-14.pt"

echo ""
echo "=== Step 5: Upload parquets ==="
rclone copy /tmp/aesthetic-hub-output/global-uhd          chi_tacc:${CONTAINER}/datasets/global-uhd          --progress
rclone copy /tmp/aesthetic-hub-output/global-flickr       chi_tacc:${CONTAINER}/datasets/global-flickr       --progress
rclone copy /tmp/aesthetic-hub-output/personalized-flickr chi_tacc:${CONTAINER}/datasets/personalized-flickr --progress

echo ""
echo "=== Step 6: Upload raw metadata files ==="
rclone copy /data/uhd-iqa/uhd-iqa-metadata.csv                           chi_tacc:${CONTAINER}/raw-data/uhd-iqa/
rclone copy /data/flickr-aes/FLICKR-AES_image_score.txt                  chi_tacc:${CONTAINER}/raw-data/flickr-aes/
rclone copy /data/flickr-aes/FLICKR-AES_image_labeled_by_each_worker.csv chi_tacc:${CONTAINER}/raw-data/flickr-aes/

echo ""
echo "=== Step 7: Upload images (~12GB, slow) ==="
rclone copy /data/uhd-iqa/training   chi_tacc:${CONTAINER}/raw-data/uhd-iqa/images  --progress
rclone copy /data/uhd-iqa/validation chi_tacc:${CONTAINER}/raw-data/uhd-iqa/images  --progress
rclone copy /data/uhd-iqa/test       chi_tacc:${CONTAINER}/raw-data/uhd-iqa/images  --progress
rclone copy /data/flickr-aes/40K     chi_tacc:${CONTAINER}/raw-data/flickr-aes/images --progress

echo ""
echo "=== Step 8: Write and upload manifests + dataset cards ==="
python3 write_metadata.py
rclone copy /tmp/aesthetic-hub-metadata chi_tacc:${CONTAINER} --progress

echo ""
echo "============================================================"
echo "INGESTION COMPLETE"
echo "============================================================"