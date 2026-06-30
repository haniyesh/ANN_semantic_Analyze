from datetime import datetime, timezone


class NewsCache:
    """In-memory TTL cache for news items.

    Attribute names are normalized to ``store`` / ``expiry`` and used
    consistently across every method (the previous version mixed
    ``store`` and ``_store``, raising AttributeError on get/delete/size).
    """

    def __init__(self):
        self.store = {}
        self.expiry = {}

    def set(self, key: str, value, ttl: int = 60):
        """Save a value with a key. ``ttl`` = seconds before it expires."""
        self.store[key] = value
        self.expiry[key] = datetime.now(timezone.utc).timestamp() + ttl

    def get(self, key: str):
        """Get a value by key. Returns None if missing or expired."""
        if key not in self.store:
            return None
        if self.is_expired(key):
            self.delete(key)
            return None
        return self.store[key]

    def delete(self, key: str):
        """Remove one item from cache."""
        self.store.pop(key, None)
        self.expiry.pop(key, None)

    def clear(self):
        """Wipe everything from cache."""
        self.store.clear()
        self.expiry.clear()

    def is_expired(self, key: str) -> bool:
        """Check if a key has passed its expiry time."""
        if key not in self.expiry:
            return True
        return datetime.now(timezone.utc).timestamp() > self.expiry[key]

    def size(self) -> int:
        """How many items are currently in cache."""
        return len(self.store)


# Backwards-compatible alias (old typo'd class name).
NewsCashe = NewsCache

cache = NewsCache()
