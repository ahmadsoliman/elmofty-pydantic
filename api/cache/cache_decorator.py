from functools import wraps
from api.cache.redis_manager import get_redis
import json
import hashlib


def cache_response(ttl: int = 3600, key_of_arg=str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            redis = get_redis()

            # Create cache key
            key_parts = (
                [func.__name__]
                + [key_of_arg(arg) for arg in args]
                + [f"{k}={v}" for k, v in kwargs.items()]
            )

            cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()

            # Check cache
            cached = redis.get(cache_key)

            if cached:
                return json.loads(cached)

            # Execute and cache
            result = await func(*args, **kwargs)
            redis.set(cache_key, json.dumps(result), ex=ttl)
            return result

        return wrapper

    return decorator
