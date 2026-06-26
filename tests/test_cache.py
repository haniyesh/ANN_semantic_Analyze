"""Round-trip tests for the in-memory TTL cache.

These would have caught the `_store` vs `store` AttributeError that made
get/delete/size unusable.
"""
import time

from storage.cache import NewsCache, cache as global_cache


def test_set_get_roundtrip():
    c = NewsCache()
    c.set("k", {"v": 1}, ttl=60)
    assert c.get("k") == {"v": 1}


def test_size_and_delete():
    c = NewsCache()
    c.set("a", 1)
    c.set("b", 2)
    assert c.size() == 2
    c.delete("a")
    assert c.size() == 1
    assert c.get("a") is None


def test_expiry():
    c = NewsCache()
    c.set("k", "v", ttl=0)
    time.sleep(0.01)
    assert c.get("k") is None
    assert c.size() == 0  # expired key auto-removed on get


def test_clear():
    c = NewsCache()
    c.set("a", 1)
    c.clear()
    assert c.size() == 0


def test_global_instance_usable():
    global_cache.clear()
    global_cache.set("x", 42)
    assert global_cache.get("x") == 42
