import os
from pydantic.v1 import BaseSettings, Field
from loguru import logger


class Settings(BaseSettings):
    POSTGRES_DB: str = Field("test_db", env="POSTGRES_DB")
    POSTGRES_USER: str = Field("test_user", env="POSTGRES_USER")
    POSTGRES_PASSWORD: str = Field("test_pass", env="POSTGRES_PASSWORD")

    is_docker: bool = Field(default_factory=lambda: os.environ.get("DOCKER_ENV") == "1")

    database_url: str = Field(..., env="DATABASE_URL")
    redis_url: str = Field(..., env="REDIS_URL")
    flower_basic_auth: str = Field(..., env="FLOWER_BASIC_AUTH")
    broker_url: str = Field(..., env="BROKER_URL")

    class Config:
        env_file = ".env"

    @classmethod
    def create(cls):
        """Create instance and dynamically set database and redis URLs."""
        instance = cls()
        # Set the correct database URL without the +asyncpg
        instance.database_url = (
            f"postgres://{instance.POSTGRES_USER}:{instance.POSTGRES_PASSWORD}@"
            f"{'localhost' if not instance.is_docker else 'postgres'}:5432/{instance.POSTGRES_DB}"
        )
        instance.redis_url = (
            f"redis://{'localhost' if not instance.is_docker else 'redis'}:6379/0"
        )
        instance.broker_url = instance.redis_url
        return instance


# Instantiate Settings
settings = Settings.create()

if __name__ == "__main__":
    logger.info(f"Settings: {settings.dict()}")
