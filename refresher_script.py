import os
import json
import csv
import time
import hashlib
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
from dotenv import load_dotenv
import chromadb

# ———————————————
# Configuration
# ———————————————
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client  = chromadb.PersistentClient(path="chroma")
QUESTION_STORE = chroma_client.get_or_create_collection(name="quizme_test_1")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CENTRAL_JSON = os.path.join(SCRIPT_DIR, "all_questions.json")
CENTRAL_CSV  = os.path.join(SCRIPT_DIR, "all_questions.csv")

# ———————————————
# Safe-load central JSON
# ———————————————
central_data = []
if os.path.exists(CENTRAL_JSON):
    raw = open(CENTRAL_JSON, "r", encoding="utf-8").read().strip()
    if raw:
        try:
            central_data = json.loads(raw)
        except json.JSONDecodeError:
            print(f"⚠️ Warning: {CENTRAL_JSON} is invalid JSON; starting fresh.")
            central_data = []
    else:
        # file exists but is empty
        central_data = []

# Build set for deduplication
central_texts = {q["question"].lower().strip() for q in central_data}

# Prepare CSV header if needed
csv_fields = [
    "Index", "Question", "Difficulty", "Correct Answer",
    "Option 1", "Option 2", "Option 3", "Option 4", "Source"
]
if not os.path.exists(CENTRAL_CSV):
    with open(CENTRAL_CSV, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=csv_fields)
        writer.writeheader()

# Start index at existing count
index_counter = len(central_data)

# ———————————————
# Helpers
# ———————————————
def generate_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_embedding(text: str):
    try:
        resp = openai_client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )
        return resp.data[0].embedding
    except Exception as e:
        print(f"❌ Embed error for '{text[:30]}...': {e}")
        return None

# def is_similar(collection, embedding, threshold: float = 0.1) -> bool:
#     if embedding is None:
#         return False
#     emb = embedding[0] if isinstance(embedding, list) and isinstance(embedding[0], list) else embedding
#     res = collection.query(query_embeddings=[emb], n_results=3)
#     for dist, doc in zip(res["distances"][0], res["documents"][0]):
#         if dist < (1 - threshold):
#             print(f"⚠️ Similar found: {doc} (dist={dist})")
#             return True
#     return False

def is_similar(embedding, collection, similarity_threshold=0.1):
    """
    Checks if the given embedding is semantically similar to any existing item in the collection.
    Args:
        embedding: The embedding of the new question.
        collection: The ChromaDB collection to check against.
        similarity_threshold: Cosine similarity score above which items are considered similar.
                              (e.g., 0.9 means >90% similar).
    Returns:
        True if a similar item is found, False otherwise.
    """
    if embedding is None:
        return False

    # Ensure embedding is in the correct format (list of floats)
    embedding_values = embedding[0] if isinstance(embedding, list) and isinstance(embedding[0], list) else embedding

    # Crucial: Adjust n_results.
    # Set to a large number to ensure you check against enough potential similar items.
    # If your collection is very large, choose a number that balances recall and performance (e.g., 500, 1000).
    # If the collection is small (e.g., up to ~5k-10k), querying all might be fine.
    # We'll use count() to get the current number of items.
    current_collection_size = collection.count()
    # For small collections, query all. For larger, cap at a reasonable number.
    n_results_to_query = min(current_collection_size, 50000) # Query up to 500 results, or less if collection is smaller
    if n_results_to_query == 0: # If collection is empty, no similar items can exist.
        return False

    results = collection.query(
        query_embeddings=[embedding_values],
        n_results=n_results_to_query,
        include=['distances', 'documents'] # Make sure to include distances for the check
    )

    # Check if any results were actually returned
    if not results or not results["distances"] or not results["distances"][0]:
        return False # No results found, so no similar items

    # Iterate through the returned top N results
    for dist, doc in zip(results["distances"][0], results["documents"][0]):
        # Assuming `dist` is Cosine Distance (1 - Cosine Similarity)
        # So, `dist < (1 - similarity_threshold)` means `similarity > similarity_threshold`
        if dist < (1 - similarity_threshold):
            # print(f"  --> Skipping: Similar item found: '{doc}' (distance: {dist:.4f}, required_dist_lt: {1-similarity_threshold:.4f})")
            return True # Found a similar item above the threshold, so skip appending

    return False # No similar items found above the threshold

def add_to_chroma(collection, docs, embeddings, ids, metadata) -> bool:
    try:
        collection.upsert(documents=docs, embeddings=embeddings, ids=ids, metadatas=metadata)
        return True
    except Exception as e:
        print(f"❌ Chroma upsert failed: {e}")
        return False

# ———————————————
# Main loop
# ———————————————
BASE = os.getcwd()

for folder, _, files in os.walk(BASE):
    if "previous_questions.json" not in files:
        continue

    # pj = os.path.join(folder, "previous_questions.json")
    pj = os.path.join(folder, "all_questions.json")

    try:
        raw = open(pj, "r", encoding="utf-8").read().strip()
        if not raw:
            print(f"⚠️ Skipping empty file: {pj}")
            continue
        questions = json.loads(raw)
    except Exception as e:
        print(f"❌ Could not load {pj}: {e}")
        continue

    # 1) Central aggregation (JSON + CSV)
    csv_rows = []
    for q in questions:
        q_text = q.get("question", "").strip()
        if not q_text:
            continue
        key = q_text.lower()
        if key in central_texts:
            continue

        # mark and append
        central_texts.add(key)
        central_data.append(q)

        opts = q.get("options", {})
        csv_rows.append({
            "question": q_text,
            "difficulty": q.get("difficulty", "").strip(),
            "answer": q.get("answer", "").strip(),
            "option 1": opts.get("A", "").strip(),
            "option 2": opts.get("B", "").strip(),
            "option 3": opts.get("C", "").strip(),
            "option 4": opts.get("D", "").strip(),
            "topic": q.get("topic", "").strip(),
        })
        index_counter += 1

    # write central JSON & CSV if new rows
    if csv_rows:
        with open(CENTRAL_JSON, "w", encoding="utf-8") as f:
            json.dump(central_data, f, indent=2, ensure_ascii=False)
        with open(CENTRAL_CSV, "a", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=csv_fields)
            writer.writerows(csv_rows)
        print(f"📦 Appended {len(csv_rows)} questions from {pj}")

    # 2) ChromaDB ingestion
    for q in questions:
        q_text = q.get("question", "").strip()
        if not q_text:
            continue
        # Only ingest if we just added it centrally
        if q_text.lower() not in central_texts:
            continue

        emb = get_embedding(q_text)
        if is_similar(emb, QUESTION_STORE):
            print(f"⚠️ Skipping: Similar question already exists in Chroma: {q_text}")
            continue

        meta = [{"source": folder, "timestamp": time.time()}]
        if add_to_chroma(
            collection=QUESTION_STORE,
            docs=[q_text],
            embeddings=[emb],
            ids=[generate_hash(q_text)],
            metadata=meta
        ):
            print(f"✅ Ingested to Chroma: {q_text}")

print("🎉 All done.")
