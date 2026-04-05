import uuid
import logging
from datetime import datetime, timezone

from fastapi import FastAPI, File, Form, UploadFile, Depends, HTTPException
from sqlalchemy.orm import Session

from . import models, database, schemas
from .clip_encoder import encode, CLIP_VERSION, load_model
from .user_store import get_user_state
from .inference_log_writer import inference_log_writer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Aesthetic Hub - Online Feature Service")

database.Base.metadata.create_all(bind=database.engine)

_stats = {"requests": 0, "cold_start": 0, "errors": 0}


@app.on_event("startup")
async def startup():
    await inference_log_writer.start()
    load_model()
    logger.info("feature service ready")


@app.on_event("shutdown")
async def shutdown():
    await inference_log_writer.stop()


@app.get("/health")
async def health():
    return {
        "status":     "healthy",
        "clip_model": CLIP_VERSION,
        "stats":      _stats,
        "log_writer": inference_log_writer.get_stats(),
    }


@app.post("/score-image", response_model=schemas.ScoreImageResponse)
async def score_image(
    image:   UploadFile = File(...),
    user_id: str        = Form(...),
    source:  str        = Form("immich_upload"),
    db:      Session    = Depends(database.get_db),
):
    request_received_at = datetime.now(timezone.utc)
    request_id          = str(uuid.uuid4())
    asset_id            = image.filename or "unknown"

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty image file")

    try:
        clip_embedding = encode(image_bytes)
    except ValueError as e:
        _stats["errors"] += 1
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        _stats["errors"] += 1
        logger.error(f"clip encoding failed: {e}")
        raise HTTPException(status_code=500, detail="embedding computation failed")

    user_embedding, is_cold_start, alpha, model_version = get_user_state(user_id, db)

    computed_at = datetime.now(timezone.utc)

    db.add(models.InferenceLog(
        request_id          = request_id,
        asset_id            = asset_id,
        user_id             = user_id,
        clip_model_version  = CLIP_VERSION,
        model_version       = model_version,
        is_cold_start       = is_cold_start,
        alpha               = alpha,
        source              = source,
        request_received_at = request_received_at,
        computed_at         = computed_at,
    ))
    db.commit()

    await inference_log_writer.write({
        "request_id":          request_id,
        "asset_id":            asset_id,
        "user_id":             user_id,
        "clip_model_version":  CLIP_VERSION,
        "model_version":       model_version,
        "is_cold_start":       is_cold_start,
        "alpha":               alpha,
        "source":              source,
        "request_received_at": request_received_at.isoformat(),
        "computed_at":         computed_at.isoformat(),
    })

    _stats["requests"] += 1
    if is_cold_start:
        _stats["cold_start"] += 1

    logger.info(
        f"scored: user={user_id[:8]} asset={asset_id} "
        f"cold_start={is_cold_start} alpha={alpha:.2f} "
        f"latency={(computed_at - request_received_at).total_seconds()*1000:.0f}ms"
    )

    return schemas.ScoreImageResponse(
        request_id          = request_id,
        asset_id            = asset_id,
        user_id             = user_id,
        clip_embedding      = clip_embedding,
        user_embedding      = user_embedding,
        is_cold_start       = is_cold_start,
        alpha               = alpha,
        model_version       = model_version,
        clip_model_version  = CLIP_VERSION,
        request_received_at = request_received_at.isoformat(),
        computed_at         = computed_at.isoformat(),
    )
