"""Runtime bot settings with MongoDB persistence."""

from typing import Any, Dict, Optional

from botspot.components.data.mongo_database import get_database
from loguru import logger

# Default settings
DEFAULTS = {
    "show_cost_info": False,
}


class BotSettings:
    """Manages runtime bot settings with MongoDB persistence."""

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._db = None

    @property
    def db(self):
        if self._db is None:
            self._db = get_database()
        return self._db

    @property
    def collection(self):
        return self.db.get_collection("bot_settings")

    async def get(self, key: str) -> Any:
        """Get a setting value."""
        if key in self._cache:
            return self._cache[key]

        try:
            doc = await self.collection.find_one({"_id": key})
            if doc and "value" in doc:
                self._cache[key] = doc["value"]
                return doc["value"]
        except Exception as e:
            logger.warning(f"Error loading setting {key}: {e}")

        return DEFAULTS.get(key)

    async def set(self, key: str, value: Any) -> bool:
        """Set a setting value."""
        try:
            await self.collection.update_one(
                {"_id": key}, {"$set": {"value": value}}, upsert=True
            )
            self._cache[key] = value
            logger.info(f"Setting {key} = {value}")
            return True
        except Exception as e:
            logger.error(f"Error saving setting {key}: {e}")
            return False

    async def toggle(self, key: str) -> bool:
        """Toggle a boolean setting. Returns new value."""
        current = await self.get(key)
        new_value = not bool(current)
        await self.set(key, new_value)
        return new_value

    async def get_all(self) -> Dict[str, Any]:
        """Get all settings with their current values."""
        result = dict(DEFAULTS)
        try:
            async for doc in self.collection.find({}):
                result[doc["_id"]] = doc["value"]
        except Exception as e:
            logger.warning(f"Error loading all settings: {e}")
        return result


# Singleton instance
_settings: Optional[BotSettings] = None


def get_bot_settings() -> BotSettings:
    """Get the bot settings instance."""
    global _settings
    if _settings is None:
        _settings = BotSettings()
    return _settings
