from dotenv import load_dotenv
import os
from agents.qna_agent import run_qna_and_openai

load_dotenv()

vector_store_path = "C:\\Users\\FJ138WZ\\OneDrive - EY\\Documents\\test DocGen\\chroma"
collection_name = "doc_embeddings"

resource_name = os.getenv("resource_name")
api_key = os.getenv("api_key")
deployment_name = os.getenv("deployment_name")


if __name__ == "__main__":
    query = "OCR"
    run_qna_and_openai(query)