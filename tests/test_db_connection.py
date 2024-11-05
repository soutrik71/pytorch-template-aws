import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.db.database import get_db


@pytest.mark.asyncio
async def test_database_connection():
    # Use the get_db dependency directly for testing
    async for session in get_db():
        assert isinstance(
            session, AsyncSession
        ), "Session is not an instance of AsyncSession"

        # Check if the session can execute a simple query
        result = await session.execute(text("SELECT 1"))
        assert result.scalar() == 1, "Database did not return expected result"
