"""
Persistent access control for friends and admins using MongoDB.

This module provides functionality to:
1. Load friends and admins from MongoDB (if available)
2. Fallback to environment variables if MongoDB is not available
3. Add/remove friends dynamically via admin commands
4. Persist changes across bot restarts
"""

from typing import List, Optional

from botspot.components.data.mongo_database import get_database
from botspot.core.dependency_manager import get_dependency_manager
from loguru import logger


class PersistentAccessControl:
    """Manages persistent friends and admins lists with MongoDB support."""

    COLLECTION_NAME = "access_control"
    FRIENDS_KEY = "friends"
    ADMINS_KEY = "admins"

    def __init__(self):
        self._db = None
        self._mongo_available = None
        self._friends_cache: Optional[List[str]] = None
        self._admins_cache: Optional[List[str]] = None

    @property
    def db(self):
        """Get database connection, checking availability."""
        if self._db is None:
            try:
                self._db = get_database()
                self._mongo_available = True
            except Exception as e:
                logger.warning(
                    f"MongoDB not available, will use environment variables only: {e}"
                )
                self._mongo_available = False
                self._db = None
        return self._db

    @property
    def mongo_available(self) -> bool:
        """Check if MongoDB is available."""
        if self._mongo_available is None:
            _ = self.db  # Trigger connection check
        return self._mongo_available or False

    async def _get_from_db(self, key: str) -> Optional[List[str]]:
        """Get a list from MongoDB."""
        if not self.mongo_available:
            return None

        try:
            doc = await self.db[self.COLLECTION_NAME].find_one({"_id": key})
            if doc and "values" in doc:
                logger.debug(f"Loaded {key} from MongoDB: {doc['values']}")
                return doc["values"]
        except Exception as e:
            logger.error(f"Error loading {key} from MongoDB: {e}")
        return None

    async def _save_to_db(self, key: str, values: List[str]) -> bool:
        """Save a list to MongoDB."""
        if not self.mongo_available:
            return False

        try:
            await self.db[self.COLLECTION_NAME].update_one(
                {"_id": key}, {"$set": {"values": values}}, upsert=True
            )
            logger.info(f"Saved {key} to MongoDB: {values}")
            return True
        except Exception as e:
            logger.error(f"Error saving {key} to MongoDB: {e}")
            return False

    def _get_from_env(self, key: str) -> List[str]:
        """Get a list from environment variables via botspot settings."""
        deps = get_dependency_manager()
        if key == self.FRIENDS_KEY:
            return deps.botspot_settings.friends
        elif key == self.ADMINS_KEY:
            return deps.botspot_settings.admins
        return []

    async def get_friends(self) -> List[str]:
        """
        Get friends list, preferring MongoDB over environment variables.

        Returns:
            List of friend usernames/IDs
        """
        # Return cached value if available
        if self._friends_cache is not None:
            return self._friends_cache

        # Try to load from MongoDB
        friends_from_db = await self._get_from_db(self.FRIENDS_KEY)

        if friends_from_db is not None:
            # MongoDB has data, use it and ignore env
            logger.info(
                f"Loaded {len(friends_from_db)} friends from MongoDB, ignoring environment variables"
            )
            if len(friends_from_db) > 0:
                logger.warning(
                    "⚠️  Friends list loaded from MongoDB. Environment variable BOTSPOT_FRIENDS_STR will be ignored."
                )
            self._friends_cache = friends_from_db
            return friends_from_db

        # No MongoDB data, initialize from env
        friends_from_env = self._get_from_env(self.FRIENDS_KEY)
        logger.info(
            f"No friends data in MongoDB, initializing from environment: {friends_from_env}"
        )

        # Save to MongoDB if available
        if self.mongo_available and friends_from_env:
            await self._save_to_db(self.FRIENDS_KEY, friends_from_env)

        self._friends_cache = friends_from_env
        return friends_from_env

    async def get_admins(self) -> List[str]:
        """
        Get admins list, preferring MongoDB over environment variables.

        Returns:
            List of admin usernames/IDs
        """
        # Return cached value if available
        if self._admins_cache is not None:
            return self._admins_cache

        # Try to load from MongoDB
        admins_from_db = await self._get_from_db(self.ADMINS_KEY)

        if admins_from_db is not None:
            # MongoDB has data, use it and ignore env
            logger.info(
                f"Loaded {len(admins_from_db)} admins from MongoDB, ignoring environment variables"
            )
            if len(admins_from_db) > 0:
                logger.warning(
                    "⚠️  Admins list loaded from MongoDB. Environment variable BOTSPOT_ADMINS_STR will be ignored."
                )
            self._admins_cache = admins_from_db
            return admins_from_db

        # No MongoDB data, initialize from env
        admins_from_env = self._get_from_env(self.ADMINS_KEY)
        logger.info(
            f"No admins data in MongoDB, initializing from environment: {admins_from_env}"
        )

        # Save to MongoDB if available
        if self.mongo_available and admins_from_env:
            await self._save_to_db(self.ADMINS_KEY, admins_from_env)

        self._admins_cache = admins_from_env
        return admins_from_env

    async def add_friend(self, username: str) -> bool:
        """
        Add a friend to the friends list.

        Args:
            username: Username or user ID to add (will be normalized)

        Returns:
            True if successfully added, False otherwise
        """
        # Normalize username (ensure @ prefix for usernames)
        if username.isdigit():
            # It's a user ID, keep as is
            normalized = username
        else:
            # It's a username, ensure @ prefix
            normalized = username if username.startswith("@") else f"@{username}"

        friends = await self.get_friends()

        # Check if already in list
        if normalized in friends:
            logger.info(f"Friend {normalized} already in list")
            return False

        # Add to list
        friends.append(normalized)
        self._friends_cache = friends

        # Save to MongoDB if available
        if self.mongo_available:
            success = await self._save_to_db(self.FRIENDS_KEY, friends)
            if success:
                logger.info(f"Added friend {normalized} to MongoDB")
            return success
        else:
            logger.warning(
                f"Added friend {normalized} to in-memory cache only (MongoDB not available)"
            )
            return True

    async def remove_friend(self, username: str) -> bool:
        """
        Remove a friend from the friends list.

        Args:
            username: Username or user ID to remove

        Returns:
            True if successfully removed, False if not found
        """
        # Normalize username
        if username.isdigit():
            normalized = username
        else:
            normalized = username if username.startswith("@") else f"@{username}"

        friends = await self.get_friends()

        # Check if in list
        if normalized not in friends:
            logger.info(f"Friend {normalized} not in list")
            return False

        # Remove from list
        friends.remove(normalized)
        self._friends_cache = friends

        # Save to MongoDB if available
        if self.mongo_available:
            success = await self._save_to_db(self.FRIENDS_KEY, friends)
            if success:
                logger.info(f"Removed friend {normalized} from MongoDB")
            return success
        else:
            logger.warning(
                f"Removed friend {normalized} from in-memory cache only (MongoDB not available)"
            )
            return True

    async def add_admin(self, username: str) -> bool:
        """
        Add an admin to the admins list.

        Args:
            username: Username or user ID to add (will be normalized)

        Returns:
            True if successfully added, False otherwise
        """
        # Normalize username
        if username.isdigit():
            normalized = username
        else:
            normalized = username if username.startswith("@") else f"@{username}"

        admins = await self.get_admins()

        # Check if already in list
        if normalized in admins:
            logger.info(f"Admin {normalized} already in list")
            return False

        # Add to list
        admins.append(normalized)
        self._admins_cache = admins

        # Save to MongoDB if available
        if self.mongo_available:
            success = await self._save_to_db(self.ADMINS_KEY, admins)
            if success:
                logger.info(f"Added admin {normalized} to MongoDB")
            return success
        else:
            logger.warning(
                f"Added admin {normalized} to in-memory cache only (MongoDB not available)"
            )
            return True

    async def remove_admin(self, username: str) -> bool:
        """
        Remove an admin from the admins list.

        Args:
            username: Username or user ID to remove

        Returns:
            True if successfully removed, False if not found
        """
        # Normalize username
        if username.isdigit():
            normalized = username
        else:
            normalized = username if username.startswith("@") else f"@{username}"

        admins = await self.get_admins()

        # Check if in list
        if normalized not in admins:
            logger.info(f"Admin {normalized} not in list")
            return False

        # Remove from list
        admins.remove(normalized)
        self._admins_cache = admins

        # Save to MongoDB if available
        if self.mongo_available:
            success = await self._save_to_db(self.ADMINS_KEY, admins)
            if success:
                logger.info(f"Removed admin {normalized} from MongoDB")
            return success
        else:
            logger.warning(
                f"Removed admin {normalized} from in-memory cache only (MongoDB not available)"
            )
            return True


# Global instance
_access_control = PersistentAccessControl()


async def get_friends() -> List[str]:
    """Get the current friends list."""
    return await _access_control.get_friends()


async def get_admins() -> List[str]:
    """Get the current admins list."""
    return await _access_control.get_admins()


async def add_friend(username: str) -> bool:
    """Add a friend to the friends list."""
    return await _access_control.add_friend(username)


async def remove_friend(username: str) -> bool:
    """Remove a friend from the friends list."""
    return await _access_control.remove_friend(username)


async def add_admin(username: str) -> bool:
    """Add an admin to the admins list."""
    return await _access_control.add_admin(username)


async def remove_admin(username: str) -> bool:
    """Remove an admin from the admins list."""
    return await _access_control.remove_admin(username)
