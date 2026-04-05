CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS model_versions (
    version_id            VARCHAR PRIMARY KEY,
    dataset_version       VARCHAR NOT NULL,
    event_cutoff          TIMESTAMPTZ NOT NULL,
    train_row_count       INTEGER,
    val_row_count         INTEGER,
    unique_users          INTEGER,
    git_sha_training      VARCHAR,
    git_sha_pipeline      VARCHAR,
    mlp_object_key        VARCHAR NOT NULL,
    embeddings_object_key VARCHAR NOT NULL,
    is_active             BOOLEAN NOT NULL DEFAULT FALSE,
    activated_at          TIMESTAMPTZ,
    deactivated_at        TIMESTAMPTZ,
    created_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_model_versions_one_active
    ON model_versions(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_model_versions_active  ON model_versions(is_active);
CREATE INDEX IF NOT EXISTS idx_model_versions_created ON model_versions(created_at);


CREATE TABLE IF NOT EXISTS user_embeddings (
    user_id       VARCHAR PRIMARY KEY,
    embedding     FLOAT4[] NOT NULL,
    model_version VARCHAR NOT NULL REFERENCES model_versions(version_id),
    updated_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_embeddings_model ON user_embeddings(model_version);


CREATE TABLE IF NOT EXISTS user_interaction_counts (
    user_id           VARCHAR PRIMARY KEY,
    interaction_count INT DEFAULT 0,
    first_seen_at     TIMESTAMPTZ DEFAULT NOW(),
    updated_at        TIMESTAMPTZ DEFAULT NOW()
);


CREATE TABLE IF NOT EXISTS interaction_events (
    event_id      VARCHAR PRIMARY KEY,
    asset_id      VARCHAR NOT NULL,
    user_id       VARCHAR NOT NULL,
    event_type    VARCHAR(20) NOT NULL,
    session_id    VARCHAR NOT NULL,
    label         FLOAT NOT NULL,
    source        VARCHAR(32) NOT NULL DEFAULT 'holdout_simulation',
    model_version VARCHAR REFERENCES model_versions(version_id),
    is_cold_start BOOLEAN NOT NULL DEFAULT FALSE,
    alpha         FLOAT,
    generator_run VARCHAR,
    event_time    TIMESTAMPTZ NOT NULL,
    ingested_at   TIMESTAMPTZ DEFAULT NOW(),
    deleted_at    TIMESTAMPTZ,
    CHECK (event_type IN ('favorite', 'album_add', 'download',
                          'share', 'view_expanded', 'archive', 'delete')),
    CHECK (label >= 0.0 AND label <= 1.0),
    CHECK (source IN ('immich_upload', 'holdout_simulation'))
);

CREATE INDEX IF NOT EXISTS idx_interaction_events_user    ON interaction_events(user_id);
CREATE INDEX IF NOT EXISTS idx_interaction_events_asset   ON interaction_events(asset_id);
CREATE INDEX IF NOT EXISTS idx_interaction_events_time    ON interaction_events(event_time);
CREATE INDEX IF NOT EXISTS idx_interaction_events_type    ON interaction_events(event_type);
CREATE INDEX IF NOT EXISTS idx_interaction_events_model   ON interaction_events(model_version);
CREATE INDEX IF NOT EXISTS idx_interaction_events_session ON interaction_events(session_id);
CREATE INDEX IF NOT EXISTS idx_interaction_events_source  ON interaction_events(source);
CREATE INDEX IF NOT EXISTS idx_interaction_events_deleted ON interaction_events(deleted_at);


CREATE TABLE IF NOT EXISTS inference_log (
    request_id           VARCHAR PRIMARY KEY,
    asset_id             VARCHAR NOT NULL,
    user_id              VARCHAR NOT NULL,
    clip_model_version   VARCHAR NOT NULL,
    model_version        VARCHAR REFERENCES model_versions(version_id),
    is_cold_start        BOOLEAN NOT NULL DEFAULT FALSE,
    alpha                FLOAT,
    source               VARCHAR(32) NOT NULL DEFAULT 'immich_upload',
    request_received_at  TIMESTAMPTZ NOT NULL,
    computed_at          TIMESTAMPTZ DEFAULT NOW(),
    CHECK (source IN ('immich_upload', 'holdout_simulation'))
);

CREATE INDEX IF NOT EXISTS idx_inference_log_user  ON inference_log(user_id);
CREATE INDEX IF NOT EXISTS idx_inference_log_asset ON inference_log(asset_id);
CREATE INDEX IF NOT EXISTS idx_inference_log_model ON inference_log(model_version);
CREATE INDEX IF NOT EXISTS idx_inference_log_time  ON inference_log(computed_at);


CREATE TABLE IF NOT EXISTS aesthetic_scores (
    asset_id             VARCHAR NOT NULL,
    user_id              VARCHAR NOT NULL,
    score                FLOAT NOT NULL,
    model_version        VARCHAR REFERENCES model_versions(version_id),
    is_cold_start        BOOLEAN NOT NULL DEFAULT FALSE,
    alpha                FLOAT,
    inference_request_id VARCHAR REFERENCES inference_log(request_id),
    source               VARCHAR(32) NOT NULL DEFAULT 'immich_upload',
    scored_at            TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (asset_id, user_id),
    CHECK (score >= 0.0 AND score <= 1.0),
    CHECK (source IN ('immich_upload', 'holdout_simulation'))
);

CREATE INDEX IF NOT EXISTS idx_aesthetic_scores_user  ON aesthetic_scores(user_id);
CREATE INDEX IF NOT EXISTS idx_aesthetic_scores_asset ON aesthetic_scores(asset_id);
CREATE INDEX IF NOT EXISTS idx_aesthetic_scores_model ON aesthetic_scores(model_version);
CREATE INDEX IF NOT EXISTS idx_aesthetic_scores_time  ON aesthetic_scores(scored_at);
