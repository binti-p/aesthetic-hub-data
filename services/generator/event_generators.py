import httpx
import random
import logging
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

EVENT_TYPES  = ["favorite", "album_add", "download",
                "share", "view_expanded", "archive", "delete"]

EVENT_LABELS = {
    "favorite":      1.0,
    "album_add":     0.9,
    "download":      0.7,
    "share":         0.6,
    "view_expanded": 0.4,
    "archive":       0.1,
    "delete":        0.0,
}

BASE_WEIGHTS = {
    "favorite":      0.05,
    "album_add":     0.02,
    "download":      0.05,
    "share":         0.03,
    "view_expanded": 0.25,
    "archive":       0.10,
    "delete":        0.50,
}


class AestheticEventGenerators:

    def __init__(self, api_base_url: str, holdout_df, timeout: float = 30.0):
        self.api_url    = api_base_url
        self.client     = httpx.AsyncClient(timeout=timeout)
        self.holdout_df = holdout_df
        self.users:     List[str] = []
        self.images:    Dict[str, List[str]] = {}
        self.stats      = {"uploads": 0, "interactions": 0,
                           "errors": 0, "duplicates": 0}
        self.run_id     = str(uuid.uuid4())
        logger.info(f"generator run_id: {self.run_id}")

    def _score_to_weights(self, true_score: float) -> List[float]:
        w = dict(BASE_WEIGHTS)
        w["favorite"]      *= (0.2 + true_score * 3.0)
        w["album_add"]     *= (0.1 + true_score * 2.0)
        w["download"]      *= (0.2 + true_score * 2.0)
        w["share"]         *= (0.1 + true_score * 2.0)
        w["view_expanded"] *= (0.5 + true_score * 1.0)
        w["archive"]       *= (0.2 + (1 - true_score) * 2.0)
        w["delete"]        *= (0.1 + (1 - true_score) * 3.0)
        total = sum(w.values())
        return [w[e] / total for e in EVENT_TYPES]

    async def generate_upload(self, user_id: Optional[str] = None) -> bool:
        if user_id is None:
            user_id = random.choice(list(self.holdout_df["user_id"].unique()))

        user_images = self.holdout_df[self.holdout_df["user_id"] == user_id]
        if user_images.empty:
            return False

        row = user_images.sample(1).iloc[0]

        try:
            resp = await self.client.post(
                f"{self.api_url}/events/upload",
                json={
                    "user_id":       user_id,
                    "asset_id":      str(row["image_name"]),
                    "s3_url":        str(row["s3_url"]),
                    "source":        "holdout_simulation",
                    "generator_run": self.run_id,
                }
            )
            resp.raise_for_status()

            if user_id not in self.images:
                self.images[user_id] = []
            if str(row["image_name"]) not in self.images[user_id]:
                self.images[user_id].append(str(row["image_name"]))
            if user_id not in self.users:
                self.users.append(user_id)

            self.stats["uploads"] += 1
            return True

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"upload failed: {e}")
            return False

    async def generate_interaction(self) -> bool:
        if not self.users:
            return False

        user_id  = random.choice(self.users)
        assets   = self.images.get(user_id, [])
        if not assets:
            return False

        asset_id = random.choice(assets)

        row = self.holdout_df[
            (self.holdout_df["user_id"] == user_id) &
            (self.holdout_df["image_name"] == asset_id)
        ]
        true_score = float(row["score"].iloc[0]) if not row.empty else 0.5
        weights    = self._score_to_weights(true_score)
        event_type = random.choices(EVENT_TYPES, weights=weights, k=1)[0]
        label      = EVENT_LABELS[event_type]

        try:
            resp = await self.client.post(
                f"{self.api_url}/events/interaction",
                json={
                    "event_id":      str(uuid.uuid4()),
                    "asset_id":      asset_id,
                    "user_id":       user_id,
                    "event_type":    event_type,
                    "session_id":    str(uuid.uuid4()),
                    "label":         label,
                    "source":        "holdout_simulation",
                    "is_cold_start": False,
                    "generator_run": self.run_id,
                    "event_time":    datetime.now(timezone.utc).isoformat(),
                }
            )
            resp.raise_for_status()
            result = resp.json()

            if result.get("status") == "duplicate":
                self.stats["duplicates"] += 1
            else:
                self.stats["interactions"] += 1

            return True

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"interaction failed: {e}")
            return False

    def print_stats(self):
        logger.info("=" * 50)
        logger.info("GENERATOR STATISTICS")
        logger.info(f"uploads:      {self.stats['uploads']}")
        logger.info(f"interactions: {self.stats['interactions']}")
        logger.info(f"duplicates:   {self.stats['duplicates']}")
        logger.info(f"errors:       {self.stats['errors']}")
        logger.info(f"active users: {len(self.users)}")
        logger.info("=" * 50)

    async def close(self):
        await self.client.aclose()
