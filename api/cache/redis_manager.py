import redis
from config import settings


class RedisManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=0,
                username=settings.REDIS_USER,
                password=settings.REDIS_PASSWORD,
                decode_responses=True,
            )
        return cls._instance


def get_redis():
    return RedisManager()


def check_and_delete_nonce(nonce: str) -> bool:
    redis = get_redis()
    if nonce and redis.delete(nonce):
        return True
    return False
