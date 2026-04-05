import logging
from typing import Optional, Tuple

from sqlalchemy.orm import Session

from . import models

logger         = logging.getLogger(__name__)
USER_EMB_DIM   = 64
COLD_START_TAU = 10


def get_user_state(
    user_id: str, db: Session
) -> Tuple[list, bool, float, Optional[str]]:
    count_row = db.query(models.UserInteractionCount).filter(
        models.UserInteractionCount.user_id == user_id
    ).first()
    n     = count_row.interaction_count if count_row else 0
    alpha = n / (n + COLD_START_TAU)

    emb_row = db.query(models.UserEmbedding).filter(
        models.UserEmbedding.user_id == user_id
    ).first()

    if emb_row is None:
        logger.debug(f"user {user_id} not found - cold start zeros")
        return [0.0] * USER_EMB_DIM, True, alpha, None

    return list(emb_row.embedding), False, alpha, emb_row.model_version
