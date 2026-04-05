from sqlalchemy import Column, String, Float, Boolean, DateTime, Integer, ARRAY
from sqlalchemy.sql import func
from .database import Base


class UserEmbedding(Base):
    __tablename__ = "user_embeddings"

    user_id       = Column(String, primary_key=True)
    embedding     = Column(ARRAY(Float), nullable=False)
    model_version = Column(String, nullable=False)
    updated_at    = Column(DateTime(timezone=True), server_default=func.now())


class UserInteractionCount(Base):
    __tablename__ = "user_interaction_counts"

    user_id           = Column(String, primary_key=True)
    interaction_count = Column(Integer, default=0)
    first_seen_at     = Column(DateTime(timezone=True), server_default=func.now())
    updated_at        = Column(DateTime(timezone=True), server_default=func.now())


class InferenceLog(Base):
    __tablename__ = "inference_log"

    request_id          = Column(String, primary_key=True)
    asset_id            = Column(String, nullable=False)
    user_id             = Column(String, nullable=False)
    clip_model_version  = Column(String, nullable=False)
    model_version       = Column(String)
    is_cold_start       = Column(Boolean, nullable=False, default=False)
    alpha               = Column(Float)
    source              = Column(String, nullable=False, default="immich_upload")
    request_received_at = Column(DateTime(timezone=True), nullable=False)
    computed_at         = Column(DateTime(timezone=True), server_default=func.now())
