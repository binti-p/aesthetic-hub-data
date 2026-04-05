import io
import os
import logging
from collections import defaultdict
from datetime import date
from typing import Dict, Any, List

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

PARQUET_SCHEMA = pa.schema([
    pa.field("request_id",          pa.string()),
    pa.field("asset_id",            pa.string()),
    pa.field("user_id",             pa.string()),
    pa.field("clip_model_version",  pa.string()),
    pa.field("model_version",       pa.string()),
    pa.field("is_cold_start",       pa.bool_()),
    pa.field("alpha",               pa.float32()),
    pa.field("source",              pa.string()),
    pa.field("request_received_at", pa.string()),
    pa.field("computed_at",         pa.string()),
])


class InferenceLogWriter:

    def __init__(self):
        self.container    = os.getenv("OBJSTORE_CONTAINER", "ObjStore_proj21")
        self.prefix       = "production-sim/inference-log"
        self.batch_size   = int(os.getenv("OBJSTORE_BATCH_SIZE", "50"))
        self.enabled      = True
        self.buffer: List[Dict[str, Any]] = []
        self.part_counter = defaultdict(int)
        self._s3          = None
        self.write_count  = 0
        self.flush_count  = 0
        self.error_count  = 0

    async def start(self):
        try:
            self._s3 = boto3.client(
                "s3",
                endpoint_url=os.environ.get("S3_ENDPOINT_URL", "https://chi.tacc.chameleoncloud.org:7480"),
                aws_access_key_id=os.environ["EC2_ACCESS_KEY"],
                aws_secret_access_key=os.environ["EC2_SECRET_KEY"],
            )
            self._s3.head_bucket(Bucket=self.container)
            logger.info(f"inference log writer connected: {self.container}/{self.prefix}")
        except Exception as e:
            logger.error(f"inference log writer connection failed: {e}")
            self.enabled = False

    async def stop(self):
        if self.buffer:
            self._flush()
        logger.info(f"inference log writer - writes: {self.write_count} flushes: {self.flush_count} errors: {self.error_count}")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "enabled":     self.enabled,
            "write_count": self.write_count,
            "flush_count": self.flush_count,
            "error_count": self.error_count,
            "buffer_size": len(self.buffer),
        }

    async def write(self, record: Dict[str, Any]) -> bool:
        if not self.enabled:
            return False

        self.buffer.append(record)
        self.write_count += 1

        if len(self.buffer) >= self.batch_size:
            return self._flush()

        return True

    def _flush(self) -> bool:
        if not self.buffer or not self._s3:
            return True
        try:
            today    = date.today().isoformat()
            part_num = self.part_counter[today]
            self.part_counter[today] += 1
            key      = f"{self.prefix}/date={today}/part-{part_num:04d}.parquet"

            df = pd.DataFrame(self.buffer)
            if "model_version" not in df.columns:
                df["model_version"] = None

            table = pa.Table.from_pandas(df, schema=PARQUET_SCHEMA)
            buf   = io.BytesIO()
            pq.write_table(table, buf, compression="snappy")
            buf.seek(0)

            self._s3.put_object(
                Bucket=self.container,
                Key=key,
                Body=buf.read(),
            )

            n = len(self.buffer)
            self.buffer.clear()
            self.flush_count += 1
            logger.info(f"flushed {n} inference log records -> {key}")
            return True

        except Exception as e:
            self.error_count += 1
            logger.error(f"inference log flush failed: {e}")
            return False


inference_log_writer = InferenceLogWriter()
