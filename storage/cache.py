from datetime import datetime, timedelta,timezone

class NewsCashe:
    def __init__(self):
        self.store={}
        
        self.expiry={}
    
    def set (self, key:str , value, ttl: int=60):
        """
        Save a value with a key.
        ttl = how many seconds before it expires.
        """
        self.store[key]=value
        self.expiry[key]=datetime.now(timezone.utc).timestamp()+ttl
        
    def get(self, key: str):
        """
        Get a value by key.
        Returns None if not found or expired.
        """
        if key not in self._store:
            return None

        if self.is_expired(key):
            self.delete(key)
            return None

        return self._store[key]

    # ==============================
    # 🗑️ DELETE
    # ==============================
    def delete(self, key: str):
        """
        Remove one item from cache.
        """
        self._store.pop(key, None)
        self._expiry.pop(key, None)

    # ==============================
    # 🧹 CLEAR
    # ==============================
    def clear(self):
        """
        Wipe everything from cache.
        """
        self._store.clear()
        self._expiry.clear()

    # ==============================
    # ⏰ IS EXPIRED
    # ==============================
    def is_expired(self, key: str) -> bool:
        """
        Check if a key has passed its expiry time.
        """
        if key not in self._expiry:
            return True
        return datetime.now(timezone.utc).timestamp() > self._expiry[key]

    # ==============================
    # 📊 SIZE
    # ==============================
    def size(self) -> int:
        """
        How many items are currently in cache.
        """
        return len(self._store)


# ==============================
# 🌍 GLOBAL INSTANCE
# ==============================
cache = NewsCashe()