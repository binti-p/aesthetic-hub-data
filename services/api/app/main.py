import os
import uuid
import logging
from datetime import datetime, timezone

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from . import models, database, schemas
from .objstore_writer import objstore_writer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Aesthetic Hub API")

database.Base.metadata.create_all(bind=database.engine)


@app.on_event("startup")
async def startup_event():
    await objstore_writer.start()


@app.on_event("shutdown")
async def shutdown_event():
    await objstore_writer.stop()


@app.get("/health")
async def health_check():
    return {
        "status":       "healthy",
        "writer_stats": objstore_writer.get_stats(),
    }


@app.post("/events/upload", response_model=schemas.UploadResponse)
async def record_upload(
    event: schemas.UploadEvent,
    db: Session = Depends(database.get_db)
):
    if event.source not in schemas.VALID_SOURCES:
        raise HTTPException(status_code=400, detail=f"invalid source: {event.source}")

    stmt = insert(models.UserInteractionCount).values(
        user_id=event.user_id,
        interaction_count=0,
    ).on_conflict_do_nothing(index_elements=["user_id"])
    db.execute(stmt)
    db.commit()

    logger.info(f"upload: user={event.user_id[:8]} asset={event.asset_id[:8]}")
    return {"status": "ok", "event_id": str(uuid.uuid4())}


@app.post("/events/interaction", response_model=schemas.InteractionResponse)
async def record_interaction(
    event: schemas.InteractionEvent,
    db: Session = Depends(database.get_db)
):
    if event.event_type not in schemas.EVENT_LABELS:
        raise HTTPException(status_code=400, detail=f"invalid event_type: {event.event_type}")

    if event.source not in schemas.VALID_SOURCES:
        raise HTTPException(status_code=400, detail=f"invalid source: {event.source}")

    existing = db.query(models.InteractionEvent).filter(
        models.InteractionEvent.event_id == event.event_id
    ).first()
    if existing:
        return {"status": "duplicate", "event_id": event.event_id}

    ingested_at = datetime.now(timezone.utc)

    db_event = models.InteractionEvent(
        event_id      = event.event_id,
        asset_id      = event.asset_id,
        user_id       = event.user_id,
        event_type    = event.event_type,
        session_id    = event.session_id,
        label         = event.label,
        source        = event.source,
        model_version = event.model_version,
        is_cold_start = event.is_cold_start,
        alpha         = event.alpha,
        generator_run = event.generator_run,
        event_time    = datetime.fromisoformat(event.event_time),
        ingested_at   = ingested_at,
    )
    db.add(db_event)

    stmt = insert(models.UserInteractionCount).values(
        user_id=event.user_id,
        interaction_count=1,
    ).on_conflict_do_update(
        index_elements=["user_id"],
        set_={
            "interaction_count": models.UserInteractionCount.interaction_count + 1,
            "updated_at":        ingested_at,
        }
    )
    db.execute(stmt)
    db.commit()

    await objstore_writer.write_event({
        "event_id":      event.event_id,
        "asset_id":      event.asset_id,
        "user_id":       event.user_id,
        "event_type":    event.event_type,
        "session_id":    event.session_id,
        "label":         event.label,
        "source":        event.source,
        "model_version": event.model_version,
        "is_cold_start": event.is_cold_start,
        "alpha":         event.alpha,
        "generator_run": event.generator_run,
        "event_time":    event.event_time,
        "ingested_at":   ingested_at.isoformat(),
    })

    logger.info(f"interaction: {event.event_type} label={event.label} user={event.user_id[:8]}")
    return {"status": "ok", "event_id": event.event_id}
