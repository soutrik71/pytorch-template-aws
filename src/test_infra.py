import asyncpg
import aioredis
from loguru import logger
from app.core.config import settings


async def test_postgres_connection(database_url: str):
    try:
        conn = await asyncpg.connect(database_url)
        logger.info("Successfully connected to PostgreSQL!")
        await conn.close()
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")


async def test_redis_connection(redis_url: str):
    try:
        redis = await aioredis.from_url(redis_url)
        await redis.ping()  # Send a ping to check connection
        logger.info("Successfully connected to Redis!")
        await redis.close()
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")


async def main():
    logger.info(f"Settings: {settings.dict()}")

    await test_postgres_connection(settings.database_url.replace("+asyncpg", ""))
    await test_redis_connection(settings.redis_url)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
