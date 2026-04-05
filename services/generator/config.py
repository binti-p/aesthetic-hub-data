import os


class Config:
    def __init__(self):
        self.api_url         = os.getenv("GENERATOR_API_URL", "http://localhost:8000")
        self.arrival_rate    = float(os.getenv("GENERATOR_ARRIVAL_RATE", "500.0"))
        self.initial_users   = int(os.getenv("GENERATOR_INITIAL_USERS", "5"))
        self.log_level       = os.getenv("GENERATOR_LOG_LEVEL", "INFO")
        self.request_timeout = float(os.getenv("GENERATOR_REQUEST_TIMEOUT", "30.0"))
        self.container       = os.getenv("OBJSTORE_CONTAINER", "ObjStore_proj21")
        self.holdout_parquet = os.getenv(
            "HOLDOUT_PARQUET_PATH",
            "datasets/personalized-flickr/new_user_holdout.parquet"
        )
