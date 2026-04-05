import asyncio
import io
import logging
import os
import random
import signal
import sys
import boto3
import numpy as np
import pandas as pd

from .config import Config
from .event_generators import AestheticEventGenerators


class PoissonEventGenerator:

    def __init__(self, mean_rate: float, event_callback, name: str = "Event"):
        self.mean_rate       = mean_rate
        self.event_callback  = event_callback
        self.name            = name
        self.running         = False
        self.event_count     = 0
        self.rate_per_second = mean_rate / 3600.0

    async def start(self):
        self.running = True
        logger = logging.getLogger(__name__)
        logger.info(f"started {self.name} at {self.mean_rate:.1f} events/hr")

        while self.running:
            wait_time = np.random.exponential(1.0 / self.rate_per_second)
            await asyncio.sleep(wait_time)

            if self.running:
                try:
                    await self.event_callback()
                    self.event_count += 1
                    if self.event_count % 50 == 0:
                        logger.info(f"{self.name}: {self.event_count} events generated")
                except Exception as e:
                    logger.error(f"event failed: {e}", exc_info=True)
                    await asyncio.sleep(1.0)

    def stop(self):
        self.running = False


def setup_logging(log_level: str):
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def load_holdout(config: Config) -> pd.DataFrame:
    s3 = boto3.client(
        "s3",
        endpoint_url="https://chi.tacc.chameleoncloud.org:7480",
        aws_access_key_id=os.environ["EC2_ACCESS_KEY"],
        aws_secret_access_key=os.environ["EC2_SECRET_KEY"],
    )
    buf = io.BytesIO()
    s3.download_fileobj(config.container, config.holdout_parquet, buf)
    buf.seek(0)
    df = pd.read_parquet(buf)
    logger = logging.getLogger(__name__)
    logger.info(f"loaded holdout: {df['user_id'].nunique()} users, {len(df):,} rows")
    return df


async def main():
    config = Config()
    setup_logging(config.log_level)
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("AESTHETIC HUB DATA GENERATOR")
    logger.info(f"api:          {config.api_url}")
    logger.info(f"arrival rate: {config.arrival_rate:.2f} events/hour")
    logger.info("=" * 70)

    holdout_df = load_holdout(config)

    generators = AestheticEventGenerators(
        api_base_url=config.api_url,
        holdout_df=holdout_df,
        timeout=config.request_timeout,
    )

    if config.initial_users > 0:
        all_users       = holdout_df["user_id"].unique().tolist()
        bootstrap_users = random.sample(all_users, min(config.initial_users, len(all_users)))
        logger.info(f"bootstrapping {len(bootstrap_users)} users...")
        for user_id in bootstrap_users:
            await generators.generate_upload(user_id=user_id)

    async def random_event():
        if random.random() < 0.3 or not generators.users:
            await generators.generate_upload()
        else:
            await generators.generate_interaction()

    generator     = PoissonEventGenerator(
        mean_rate=config.arrival_rate,
        event_callback=random_event,
        name="AestheticHub Generator"
    )
    shutdown_event = asyncio.Event()

    def signal_handler(sig, frame):
        logger.info(f"received signal {sig}, shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    async def print_stats_periodically():
        while not shutdown_event.is_set():
            await asyncio.sleep(300)
            if not shutdown_event.is_set():
                generators.print_stats()

    stats_task     = asyncio.create_task(print_stats_periodically())
    generator_task = asyncio.create_task(generator.start())

    try:
        logger.info("generator running. Ctrl+C to stop.")
        await shutdown_event.wait()
        generator.stop()
        stats_task.cancel()
        generator_task.cancel()
        await asyncio.wait([generator_task, stats_task], timeout=5.0)
    except Exception as e:
        logger.error(f"fatal error: {e}", exc_info=True)
    finally:
        generators.print_stats()
        await generators.close()
        logger.info("shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
