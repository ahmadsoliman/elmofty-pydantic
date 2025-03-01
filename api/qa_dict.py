import logging
from pydantic import BaseModel
import supabase

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
        try:
            supabase_client = supabase.create_client(url, key)

            qa_data = []
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

                # break  # TODO: REMOVE

            _qa_dict = {
                str(qa["id"]): QA(
                    id=str(qa["id"]), question=qa["question"], answer=qa["answer"]
                )
                for qa in qa_data
            }
        except Exception as e:
            logger.error(f"Error loading QA dictionary: {e}")
            return {}
    return _qa_dict
