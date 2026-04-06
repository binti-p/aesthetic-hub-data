import os
import logging
import sqlalchemy as sa
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    db_url = os.environ["DATABASE_URL"]
    engine = sa.create_engine(db_url)

    with engine.connect() as conn:
        # prune interaction_events older than 30 days
        result = conn.execute(sa.text("""
            DELETE FROM interaction_events
            WHERE ingested_at < NOW() - INTERVAL '30 days'
            AND deleted_at IS NULL
        """))
        conn.commit()
        logger.info(f"pruned {result.rowcount} interaction_events")

        # prune inference_log older than 30 days
        result = conn.execute(sa.text("""
            DELETE FROM inference_log
            WHERE computed_at < NOW() - INTERVAL '30 days'
        """))
        conn.commit()
        logger.info(f"pruned {result.rowcount} inference_log rows")

if __name__ == "__main__":
    main()