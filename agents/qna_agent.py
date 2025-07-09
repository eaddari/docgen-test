import os
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import AzureOpenAI

from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def run(self, query: str) -> str:
        """Process a query and return a response."""
        pass

load_dotenv()

class QnAAgent(BaseAgent):
    def __init__(self, vector_store_path, collection_name, embedding_model="all-MiniLM-L6-v2"):
        self.client = chromadb.PersistentClient(path=vector_store_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedder = SentenceTransformer(embedding_model)

    def run(self, query: str) -> str:
        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=10)
        if results["metadatas"] and results["metadatas"][0]:
            texts = [meta.get("text", "No text found.") for meta in results["metadatas"][0]]
            for i, text in enumerate(texts, 1):
                print(f"Chunk {i}: {text}")
            return "\n\n".join(texts)
        return "No relevant information found."

class OpenAIClient:
    def __init__(self):
        self.client = AzureOpenAI(
            api_version=os.getenv("API_VERSION"),
            azure_endpoint=os.getenv("ENDPOINT"),
            api_key=os.getenv("SUBSCRIPTION_KEY")
        )
        self.model = os.getenv("DEPLOYMENT")
    
    def chat(self, messages, **kwargs):
        return self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            **kwargs
        )

def run_qna_and_openai(query: str):
    vector_store_path = "C:\\Users\\FJ138WZ\\OneDrive - EY\\Documents\\test DocGen\\chroma"
    collection_name = "doc_embeddings"

    qna = QnAAgent(vector_store_path, collection_name)
    qna_response = qna.run(query)

    openai_client = OpenAIClient()
    openai_response = openai_client.chat(
        messages=[
            {"role": "system", "content": "You are an assistant that translates the queries retrieved from a vector store in natural language. "
            "You must exclusively answer based on the context given. The content is provided by a vector database that retrieves relevant information based on the query."
            "If the query is off topic from the Azure AI 102 Exam, answer with 'No relevant information found.'"},
            {"role": "user", "content": query}
        ],
        max_completion_tokens=800,
        temperature=0.1,
        top_p=0.5,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    print(f"OpenAI Response: {openai_response.choices[0].message.content}")


if __name__ == "__main__":
    pass
