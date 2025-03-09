import supabase
from supabase.lib.client_options import SyncClientOptions
from config import settings


class Database:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            url = settings.SUPABASE_VECTOR_URL
            key = settings.SUPABASE_VECTOR_KEY
            options = SyncClientOptions(
                postgrest_client_timeout=10,
                storage_client_timeout=10,
                auto_refresh_token=False,
                persist_session=False,
            )
            try:
                cls._instance = supabase.create_client(url, key, options)
            except:
                pass
        return cls._instance


def get_db():
    return Database()
