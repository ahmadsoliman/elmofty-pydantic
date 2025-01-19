import json

with open("public/qa.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)
qa_dict = {str(qa["id"]): qa for qa in qa_data}
