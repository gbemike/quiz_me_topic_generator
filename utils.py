import os
import time
import json
import csv
import hashlib
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

chroma_client = chromadb.PersistentClient(path="db")
QUESTION_STORE = chroma_client.get_or_create_collection(name="questions")

def generate_hash(content):
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def get_embedding(text):
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small",
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Failed to get embedding: {e}")
        return None

def is_similar(embedding, collection, threshold=0.9):
    # embedding = get_embedding(content)
    # print(f"embedding: {embedding}")
    if embedding is None:
        return False

    embedding_values = embedding[0] if isinstance(embedding, list) and isinstance(embedding[0], list) else embedding
    results = collection.query(query_embeddings=[embedding_values], n_results=3)

    for dist, doc in zip(results["distances"][0], results["documents"][0]):
        if dist < (1 - threshold):
            print(f"Similar item found: {doc} (distance: {dist})")
            return True
    return False


def add_to_chroma(collection, docs, embeddings, ids, metadata):
    try:
        collection.upsert(
            documents=docs,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadata,
        )
    except Exception as e:
        print(f"❌ Failed to add to ChromaDB: {e}")
        return False
