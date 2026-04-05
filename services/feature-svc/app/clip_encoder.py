import io
import os
import logging

import boto3
import clip
import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger        = logging.getLogger(__name__)
CLIP_MODEL    = "ViT-L/14"
CLIP_VERSION  = "ViT-L/14 openai/clip"
EMBEDDING_DIM = 768
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_PATH  = "/tmp/ViT-L-14.pt"
WEIGHTS_KEY   = "models/clip/ViT-L-14.pt"

_model      = None
_preprocess = None


def _download_weights():
    if os.path.exists(WEIGHTS_PATH):
        logger.info("CLIP weights already cached")
        return
    logger.info("downloading CLIP weights from object store...")
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT_URL", "https://chi.tacc.chameleoncloud.org:7480"),
        aws_access_key_id=os.environ["EC2_ACCESS_KEY"],
        aws_secret_access_key=os.environ["EC2_SECRET_KEY"],
    )
    s3.download_file("ObjStore_proj21", WEIGHTS_KEY, WEIGHTS_PATH)
    logger.info(f"CLIP weights downloaded: {os.path.getsize(WEIGHTS_PATH)/1e6:.0f}MB")


def load_model():
    global _model, _preprocess
    if _model is None:
        _download_weights()
        logger.info(f"loading CLIP {CLIP_MODEL} on {DEVICE}...")
        _model, _preprocess = clip.load(WEIGHTS_PATH, device=DEVICE)
        _model.eval()
        logger.info("CLIP loaded successfully")
    return _model, _preprocess


@torch.no_grad()
def encode(image_bytes: bytes) -> list:
    model, preprocess = load_model()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"could not decode image: {e}")

    tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    feat   = model.encode_image(tensor)
    feat   = feat / feat.norm(dim=-1, keepdim=True)
    emb    = feat.cpu().numpy().flatten().astype(np.float32)

    if emb.shape[0] != EMBEDDING_DIM:
        raise RuntimeError(f"expected {EMBEDDING_DIM}-d embedding, got {emb.shape[0]}")

    return emb.tolist()
