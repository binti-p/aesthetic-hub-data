from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ARRAY
from sqlalchemy.sql import func
from .database import Base


class ModelVersion(Base):
    __tablename__ = "model_versions"

    version_id            = Column(String, primary_key=True)
    dataset_version       = Column(String, nullable=False)
    event_cutoff          = Column(DateTime(timezone=True), nullable=False)
    train_row_count       = Column(Integer)
    val_row_count         = Column(Integer)
    unique_users          = Column(Integer)
    git_sha_training      = Column(String)
    git_sha_pipeline      = Column(String)
    mlp_object_key        = Column(String, nullable=False)
    embeddings_object_key = Column(String, nullable=False)
    is_active             = Column(Boolean, nullable=False, default=False)
    activated_at          = Column(DateTime(timezone=True))
    deactivated_at        = Column(DateTime(timezone=True))
    created_at            = Column(DateTime(timezone=True), server_default=func.now())


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


class InteractionEvent(Base):
    __tablename__ = "interaction_events"

    event_id      = Column(String, primary_key=True)
    asset_id      = Column(String, nullable=False)
    user_id       = Column(String, nullable=False)
    event_type    = Column(String, nullable=False)
    session_id    = Column(String, nullable=False)
    label         = Column(Float, nullable=False)
    source        = Column(String, nullable=False, default="holdout_simulation")
    model_version = Column(String)
    is_cold_start = Column(Boolean, nullable=False, default=False)
    alpha         = Column(Float)
    generator_run = Column(String)
    event_time    = Column(DateTime(timezone=True), nullable=False)
    ingested_at   = Column(DateTime(timezone=True), server_default=func.now())
    deleted_at    = Column(DateTime(timezone=True))


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


class AestheticScore(Base):
    __tablename__ = "aesthetic_scores"

    asset_id             = Column(String, primary_key=True)
    user_id              = Column(String, primary_key=True)
    score                = Column(Float, nullable=False)
    model_version        = Column(String)
    is_cold_start        = Column(Boolean, nullable=False, default=False)
    alpha                = Column(Float)
    inference_request_id = Column(String)
    source               = Column(String, nullable=False, default="immich_upload")
    scored_at            = Column(DateTime(timezone=True), server_default=func.now())

class Asset(Base):
    __tablename__ = "assets"

    asset_id      = Column(String, primary_key=True)
    s3_url        = Column(String, nullable=False)
    user_id       = Column(String, nullable=False)
    source        = Column(String, nullable=False, default="holdout_simulation")
    first_seen_at = Column(DateTime(timezone=True), server_default=func.now())