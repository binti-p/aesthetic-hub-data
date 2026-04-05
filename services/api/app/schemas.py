from pydantic import BaseModel, Field
from typing import Optional


EVENT_LABELS = {
    "favorite":      1.0,
    "album_add":     0.9,
    "download":      0.7,
    "share":         0.6,
    "view_expanded": 0.4,
    "archive":       0.1,
    "delete":        0.0,
}

VALID_SOURCES = {"immich_upload", "holdout_simulation"}


class UploadEvent(BaseModel):
    user_id:       str
    asset_id:      str
    s3_url:        str
    source:        str = "holdout_simulation"
    generator_run: Optional[str] = None


class InteractionEvent(BaseModel):
    event_id:      str
    asset_id:      str
    user_id:       str
    event_type:    str
    session_id:    str
    label:         float = Field(..., ge=0.0, le=1.0)
    source:        str = "holdout_simulation"
    model_version: Optional[str] = None
    is_cold_start: bool = False
    alpha:         Optional[float] = Field(None, ge=0.0, le=1.0)
    generator_run: Optional[str] = None
    event_time:    str


class UploadResponse(BaseModel):
    status:   str
    event_id: str


class InteractionResponse(BaseModel):
    status:   str
    event_id: str
