from cachetools import TTLCache


def make_cache(maxsize: int = 100, ttl: int = 300) -> TTLCache:
    return TTLCache(maxsize=maxsize, ttl=ttl)
