from pydantic import BaseModel
from typing import Optional, List


class ScoreImageResponse(BaseModel):
    request_id:          str
    asset_id:            str
    user_id:             str
    clip_embedding:      List[float]
    user_embedding:      List[float]
    is_cold_start:       bool
    alpha:               float
    model_version:       Optional[str]
    clip_model_version:  str
    request_received_at: str
    computed_at:         str
