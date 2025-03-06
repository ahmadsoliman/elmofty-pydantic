import logging
from pydantic import BaseModel
import supabase
import json
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

url: str = settings.SUPABASE_QA_URL
key: str = settings.SUPABASE_QA_KEY


class QA(BaseModel):
    id: str
    question: str
    answer: str


_qa_dict = {}


def get_qa_dict():
    global _qa_dict
    if not _qa_dict and settings.FLASK_ENV != "testing":
        qa_data = []

        # Define the file path for cached questions
        cache_file = Path("data/questions_cache.json")
        # Try to load from file
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    qa_data = json.load(f)
            except json.JSONDecodeError:
                # If file exists but is corrupted, continue to load from Supabase
                pass
        else:
            try:
                supabase_client = supabase.create_client(url, key)

                page = 0

                while True:
                    response = (
                        supabase_client.table("qas")
                        .select("*")
                        .range(page * 1000, (page + 1) * 1000 - 1)
                        .execute()
                    )
                    if not response.data:
                        break
                    qa_data.extend(response.data)
                    page += 1

                    break

                logger.debug(
                    "Loaded "
                    + str(len(qa_data))
                    + " all IslamQA questions and answers."
                )

                if settings.FLASK_ENV != "production":
                    # Save the data to the cache file
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_file, "w", newline="\n", encoding="utf-8") as f:
                        json.dump(qa_data, f, indent=2, ensure_ascii=False)

            except Exception as e:
                logger.error(f"Error loading QA dictionary: {e}")
                return {}

        _qa_dict = {
            str(qa["id"]): QA(
                id=str(qa["id"]), question=qa["question"], answer=qa["answer"]
            )
            for qa in qa_data
        }
    return _qa_dict


def get_qas(ids):
    try:
        supabase_client = supabase.create_client(url, key)

        response = supabase_client.table("qas").select("*").in_("id", ids).execute()
        if not response.data:
            return []

        return [QA(**qa) for qa in response.data]
    except Exception as e:
        logger.error(f"Error loading QAs from supabase: {e}")
        return []
