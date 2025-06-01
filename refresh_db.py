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

# --- ChromaDB Setup ---
# Assuming 'db' is your new, clean directory.
chroma_client = chromadb.PersistentClient(path="db")
QUESTION_STORE = chroma_client.get_or_create_collection(name="quiz_me_questions")
print(f"Connected to ChromaDB collection: '{QUESTION_STORE.name}' at '{chroma_client.count_collections()}'.")

# --- Embedding Model (OpenAI for generation, but local SBERT for comparison) ---
# When you query ChromaDB with an embedding, you're implicitly using the model
# that generated the embeddings already in ChromaDB.
# For direct comparison (if you were doing it outside Chroma's query), you'd use SBERT.
# However, `collection.query` uses the embeddings you provide, so OpenAI's `text-embedding-3-small`
# is what matters for the incoming `embedding` in `is_similar`.
# So, no explicit SBERT model loading here if you rely on `collection.query` to do the comparison.

# --- Utility Functions (Provided by you) ---
def generate_hash(content):
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_embedding(text):
    """
    Fetches an embedding for the given text using OpenAI API.
    Includes retry logic for robustness.
    """
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small",
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Failed to get embedding for text: '{text[:50]}...' Error: {e}")
        # Re-raise to trigger tenacity retry
        raise

def is_similar(embedding, collection, similarity_threshold):
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
    n_results_to_query = min(current_collection_size, 10000) # Query up to 500 results, or less if collection is smaller
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

def add_to_chroma(collection, docs, embeddings, ids, metadata):
    """
    Adds documents to the ChromaDB collection in a batch.
    """
    if not docs:
        return True # Nothing to add

    try:
        collection.add(
            documents=docs,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadata,
        )
        # print(f"✅ Added {len(docs)} documents to ChromaDB.")
        return True
    except Exception as e:
        print(f"❌ Failed to add batch to ChromaDB: {e}")
        return False

# --- Main Processing Logic ---

# Define your CSV files (replace with your actual paths)
csv_files = ["official_questions.csv"]

# Set your desired similarity threshold for skipping
# If a question is > 0.9 similar to an existing one, it will be skipped.
# Lower this value (e.g., 0.8) if you want to skip broader semantic duplicates.
SKIP_SIMILARITY_THRESHOLD = 0.5

# Batching variables for efficient ChromaDB additions
batch_docs = []
batch_embeddings = []
batch_ids = []
batch_metadatas = []
BATCH_SIZE = 100 # Add documents to ChromaDB in chunks of 100

total_processed_rows = 0
total_skipped_rows = 0
total_added_rows = 0

print(f"\nStarting to process CSV files. Skip threshold for similarity: >{SKIP_SIMILARITY_THRESHOLD}")

for csv_file in csv_files:
    if not os.path.exists(csv_file):
        print(f"⚠️ CSV file not found: {csv_file}. Skipping.")
        continue

    print(f"\n--- Processing '{csv_file}' ---")
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)

        question_column_index = -1
        try:
            question_column_index = header.index("Question") # Case-sensitive
        except ValueError:
            print(f"Error: 'Question' column not found in {csv_file}. Please check header.")
            continue

        for i, row in enumerate(reader):
            total_processed_rows += 1
            if not row: # Skip empty rows
                continue

            question_text = row[question_column_index].strip()
            if not question_text: # Skip rows with empty question text
                continue

            # Generate a unique ID for the question (e.g., using a hash)
            question_id = generate_hash(question_text) # Or use a combination of CSV name + row number

            # --- Check if ID already exists (for exact duplicates, faster than embedding check) ---
            # This is a pre-check before even generating an embedding
            try:
                if QUESTION_STORE.get(ids=[question_id])['ids']:
                    # print(f"  Skipping (ID exists): '{question_text[:50]}...'")
                    total_skipped_rows += 1
                    continue
            except Exception as e:
                # Handle cases where get() might fail if ID not found, etc.
                pass # Continue to embedding check if ID lookup fails or is empty

            # --- Get Embedding for the current question ---
            embedding = None
            try:
                embedding = get_embedding(question_text)
            except Exception as e:
                print(f"Skipping row {i+1} due to embedding failure: {e}")
                continue # Skip this row if embedding fails

            if embedding is None:
                total_skipped_rows += 1
                # print(f"  Skipping (embedding failed): '{question_text[:50]}...'")
                continue

            # --- Semantic Similarity Check ---
            if QUESTION_STORE.count() > 0 and is_similar(embedding, QUESTION_STORE, similarity_threshold=SKIP_SIMILARITY_THRESHOLD):
                total_skipped_rows += 1
                # print(f"  Skipping (semantically similar): '{question_text[:50]}...'")
                continue # Skip if similar to existing questions

            # If not skipped, add to batch
            batch_docs.append(question_text)
            batch_embeddings.append(embedding)
            batch_ids.append(question_id)
            batch_metadatas.append({"source_csv": csv_file, "row_number": i + 1})

            # Add to ChromaDB in batches
            if len(batch_docs) >= BATCH_SIZE:
                if add_to_chroma(QUESTION_STORE, batch_docs, batch_embeddings, batch_ids, batch_metadatas):
                    total_added_rows += len(batch_docs)
                    print(f"  Added batch of {len(batch_docs)} from '{csv_file}'. Total added: {total_added_rows}")
                # Clear batch lists after adding
                batch_docs = []
                batch_embeddings = []
                batch_ids = []
                batch_metadatas = []

# Add any remaining items in the last batch
if batch_docs:
    if add_to_chroma(QUESTION_STORE, batch_docs, batch_embeddings, batch_ids, batch_metadatas):
        total_added_rows += len(batch_docs)
        print(f"  Added final batch of {len(batch_docs)} from '{csv_file}'. Total added: {total_added_rows}")

print("\n--- Processing Complete ---")
print(f"Total rows processed from CSVs: {total_processed_rows}")
print(f"Total rows skipped (exact ID match or semantically similar): {total_skipped_rows}")
print(f"Total rows successfully added to ChromaDB: {total_added_rows}")
print(f"Final count in ChromaDB collection: {QUESTION_STORE.count()}")