import os
from supabase import create_client, Client
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(url, key)


qa_data = []
page = 0
while True:
    response = (
        supabase.table("qas")
        .select("*")
        .range(page * 1000, (page + 1) * 1000 - 1)
        .execute()
    )
    if not response.data:
        break
    qa_data.extend(response.data)
    page += 1


class QA(BaseModel):
    id: str
    question: str
    answer: str


qa_dict = {
    str(qa["id"]): QA(id=str(qa["id"]), question=qa["question"], answer=qa["answer"])
    for qa in qa_data
}
