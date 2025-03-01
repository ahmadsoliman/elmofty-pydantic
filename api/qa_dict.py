import os
import supabase
from pydantic import BaseModel

url: str = os.getenv("SUPABASE_QA_URL")
key: str = os.getenv("SUPABASE_QA_KEY")


class QA(BaseModel):
    id: str
    question: str
    answer: str


_qa_dict = {}


def get_qa_dict():
    global _qa_dict
    if not _qa_dict and os.getenv("FLASK_ENV") != "testing":
        supabase_client: supabase.Client = None
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

            _qa_dict = {
                str(qa["id"]): QA(
                    id=str(qa["id"]), question=qa["question"], answer=qa["answer"]
                )
                for qa in qa_data
            }
        except:
            return {}

    return _qa_dict
