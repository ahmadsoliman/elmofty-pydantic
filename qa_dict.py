import json
import os
from google.cloud import storage


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


# Replace with your bucket name and blob name
bucket_name = "eslam-qa-netherlands"
source_blob_name = "rendered_qa-mo3amalat.json"
destination_file_name = "public/qa.json"

if not os.path.exists(destination_file_name):
    download_blob(bucket_name, source_blob_name, destination_file_name)

with open(destination_file_name, "r", encoding="utf-8") as f:
    qa_data = json.load(f)
qa_dict = {str(qa["id"]): qa for qa in qa_data}
