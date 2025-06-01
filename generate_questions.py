# import os
# import time
# import json
# import csv
# import hashlib
# import chromadb
# from openai import OpenAI
# from dotenv import load_dotenv
# from tenacity import retry, stop_after_attempt, wait_exponential

# # from core.utils import get_embedding, add_to_chroma, is_similar, generate_hash

# load_dotenv()

# openai_api_key = os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=openai_api_key)

# chroma_client = chromadb.PersistentClient(path="chroma")
# QUESTION_STORE = chroma_client.get_or_create_collection(name="quizme_test_1")

# def generate_hash(content):
#     return hashlib.sha256(content.encode('utf-8')).hexdigest()

# def get_embedding(text):
#     try:
#         response = client.embeddings.create(
#             input=text,
#             model="text-embedding-3-small",
#         )
#         return response.data[0].embedding
#     except Exception as e:
#         print(f"❌ Failed to get embedding: {e}")
#         return None

# # def is_similar(embedding, collection, threshold=0.1):
# #     # embedding = get_embedding(content)
# #     # print(f"embedding: {embedding}")
# #     if embedding is None:
# #         return False

# #     embedding_values = embedding[0] if isinstance(embedding, list) and isinstance(embedding[0], list) else embedding
# #     results = collection.query(query_embeddings=[embedding_values], n_results=3)

# #     for dist, doc in zip(results["distances"][0], results["documents"][0]):
# #         if dist < (1 - threshold):
# #             print(f"Similar item found: {doc} (distance: {dist})")
# #             return True
# #     return False

# def is_similar(embedding, collection, similarity_threshold):
#     """
#     Checks if the given embedding is semantically similar to any existing item in the collection.
#     Args:
#         embedding: The embedding of the new question.
#         collection: The ChromaDB collection to check against.
#         similarity_threshold: Cosine similarity score above which items are considered similar.
#                               (e.g., 0.9 means >90% similar).
#     Returns:
#         True if a similar item is found, False otherwise.
#     """
#     if embedding is None:
#         return False

#     # Ensure embedding is in the correct format (list of floats)
#     embedding_values = embedding[0] if isinstance(embedding, list) and isinstance(embedding[0], list) else embedding

#     # Crucial: Adjust n_results.
#     # Set to a large number to ensure you check against enough potential similar items.
#     # If your collection is very large, choose a number that balances recall and performance (e.g., 500, 1000).
#     # If the collection is small (e.g., up to ~5k-10k), querying all might be fine.
#     # We'll use count() to get the current number of items.
#     current_collection_size = collection.count()
#     # For small collections, query all. For larger, cap at a reasonable number.
#     n_results_to_query = min(current_collection_size, 50000) # Query up to 500 results, or less if collection is smaller
#     if n_results_to_query == 0: # If collection is empty, no similar items can exist.
#         return False

#     results = collection.query(
#         query_embeddings=[embedding_values],
#         n_results=n_results_to_query,
#         include=['distances', 'documents'] # Make sure to include distances for the check
#     )

#     # Check if any results were actually returned
#     if not results or not results["distances"] or not results["distances"][0]:
#         return False # No results found, so no similar items

#     # Iterate through the returned top N results
#     for dist, doc in zip(results["distances"][0], results["documents"][0]):
#         # Assuming `dist` is Cosine Distance (1 - Cosine Similarity)
#         # So, `dist < (1 - similarity_threshold)` means `similarity > similarity_threshold`
#         if dist < (1 - similarity_threshold):
#             # print(f"  --> Skipping: Similar item found: '{doc}' (distance: {dist:.4f}, required_dist_lt: {1-similarity_threshold:.4f})")
#             return True # Found a similar item above the threshold, so skip appending

#     return False # No similar items found above the threshold


# def add_to_chroma(collection, docs, embeddings, ids, metadata):
#     try:
#         collection.upsert(
#             documents=docs,
#             embeddings=embeddings,
#             ids=ids,
#             metadatas=metadata,
#         )
#         print(f"✅ Successfully added {len(docs)} documents to ChromaDB.")
#     except Exception as e:
#         print(f"❌ Failed to add to ChromaDB: {e}")
#         return False


# # from core.utils import get_embedding, add_to_chroma, is_similar, generate_hash




# # @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
# # def generate_questions(folder_path, existing_questions, num_questions):
# #     """Generate validated trivia questions using the provided schema."""
# #     context = " > ".join(folder_path.split(os.sep)[-2:])
# #     if existing_questions:
# #         existing_qs = json.dumps(existing_questions, indent=2, ensure_ascii=False)
# #     else:
# #         existing_qs = "None yet."
# #     # existing_qs = "\n".join([f"- {q}" for q in existing_questions])
# #     # print(f"🔍 Exisiting Questions for Memory: {existing_qs}")

# #     # prompt = f"""
# #     # You are an *unparalleled* Nigerian trivia quizmaster, possessing **broad, expert, specific, and niche knowledge** about all facets of {context}. Your expertise spans across its inception, genres, historical periods, and every intricate detail concerned with it.

# #     # Your sole task is to generate **exactly {num_questions} fun, engaging, and profoundly unique multiple-choice questions** about {context}.

# #     # **CRITICAL INSTRUCTION: EVERY SINGLE QUESTION YOU GENERATE MUST BE EXTREMELY UNIQUE AND SHOW NO SIMILARITY WHATSOEVER TO ANY OTHER QUESTION, EITHER WITHIN THE CURRENT BATCH OR FROM THE PROVIDED EXISTING QUESTIONS. THIS IS PARAMOUNT.**

# #     # 📜 **STRICT RULES FOR OUTPUT:**
# #     # 1.  OUTPUT MUST BE **VALID JSON ONLY** - ABSOLUTELY NO MARKDOWN (e.g., ```json), NO COMMENTS, NO PREAMBLE, NO POSTAMBLE, NO EXTRA TEXT, NO CHAT. Just the JSON array.
# #     # 2.  JSON STRUCTURE MUST **EXACTLY MATCH** THIS SCHEMA:
# #     #     [{{
# #     #         "question": "...",
# #     #         "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
# #     #         "answer": "...",
# #     #         "difficulty": "...",
# #     #         "topic": "..."
# #     #     }}]
# #     #     (Note: The JSON schema example above should *not* be part of the actual output from the LLM, but for your internal code's prompt, it's illustrative.)

# #     # **🚫 STRICT ANTI-DUPLICATION & UNIQUENESS DIRECTIVES:**
# #     # * **DO NOT repeat these existing questions, not even with minor rephrasing:**
# #     #     {existing_qs if existing_qs else "None yet."}
# #     # * **DO NOT produce any questions that are remotely similar in meaning, concept, or structure to any existing ones, or to each other within this batch.**
# #     # * **EACH QUESTION MUST BE AN ISLAND: It should be conceptually distinct, phrased uniquely, and not overlap semantically with any other question you generate or that is provided as existing.**
# #     # * **Aim for questions that require truly unique knowledge points, not just different ways of asking the same fact.**

# #     # **📝 CONTENT GUIDELINES:**
# #     # * All questions must have 4 options (A-D).
# #     # * Difficulty levels must be evenly distributed (e.g., easy, medium, hard, expert).
# #     # * Vary question phrasing significantly. Avoid repetitive starts like 'What is...', 'Which is...', 'Who is...'. Instead, use more fill-in-the-blank, statement completion, contextual framing, scenario-based questions, or other creative types.
# #     # * Ensure absolute factual accuracy and verifiability.
# #     # - Do not output any question that you cannot verify as accurate, if you are not sure about the accuracy of a question, do not output it.
# #     # * Questions must be structured to be genuinely hard to directly Google or search for, requiring deeper understanding.
# #     # * Terms/Phrases used in questions should be absolutely consistent (e.g., 'Igbo' never written as 'Ibo').
# #     # * Integrate Nigerian English and cultural references authentically where appropriate.
# #     # * Ensure questions are highly engaging, truly fun, and suitable for a wide audience.
# #     # * **DELVE DEEP:** Go significantly in-depth and beyond mainstream, surface-level knowledge. Avoid basic facts.
# #     # * For 'anatomy' related topics, ensure questions are engaging and insightful, not just basic facts, but still accessible to a general knowledge audience, avoiding overly expert-level medical or scientific jargon unless {context} specifically demands it.
# #     # - topics should either be based on {context}
# #     # - you are genrating trivia questions, so don't make them too lengthy, as a result make sure to be token/cost conscious when generat, and avoid overly long questions or options.

# #     # **❌ STRICT PROHIBITIONS (NO EXCEPTIONS):**
# #     # * No markdown characters (e.g., \`\`\`json, \`\`\`, #, *, -, >, etc.).
# #     # * No explanations, comments, preambles, or postambles.
# #     # * No XML, YAML, or any other formats.
# #     # * No extra fields beyond the specified JSON schema.

# #     # **Output ONLY valid JSON like this:**
# #     # [{{"question": "...", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "answer": "...", "difficulty": "...", "topic": "..."}}]
# #     # """


# #     try:
# #         prompt = f"""
# #         You are an expert Nigerian trivia quizmaster with deep, niche knowledge of **{context}** — its history, culture, people, and details.

# #         🎯 **Your Task:**
# #         Generate **exactly {num_questions} multiple-choice trivia questions** (MCQs) on **{context}**. Each question must be:
# #         - Completely **unique** (no overlap in concept, phrasing, or fact with any other question — in this batch or below).
# #         - Fun, insightful, verifiable, and not easily Googleable.
# #         - Based on **distinct** knowledge points — no repetition or rephrasing of ideas.

# #         🚫 **Avoid Duplicates**:
# #         Do NOT copy, reword, or create semantically similar questions to:
# #         {existing_qs if existing_qs else "None yet."}

# #         📄 **Format (JSON ONLY — No extras):**
# #         Return output as a **JSON array** like:
# #         [{{
# #         "question": "...",
# #         "options": {{
# #             "A": "...", "B": "...", "C": "...", "D": "..."
# #         }},
# #         "answer": "...",
# #         "difficulty": "...",
# #         "topic": "..."
# #         }}]

# #         ⚠️ **Strict Rules**:
# #         - No markdown, no headers, no explanations.
# #         - 4 options (A-D) per question. Only 1 correct answer.
# #         - Difficulty must be evenly distributed: `easy`, `medium`, `hard`, `expert`.
# #         - Use varied formats (not just “What is...”, “Who...”). Include fill-in-the-blanks, scenarios, etc.
# #         - Keep language concise. Avoid long questions/options.
# #         - Use consistent terms (e.g., always “Igbo”, not “Ibo”).
# #         - Integrate Nigerian English or cultural references where it makes sense.
# #         - Avoid jargon unless the topic truly requires it.

# #         ✅ Output ONLY the JSON array. Nothing else.
# #         """
# #         print(f"📝 Full Prompt: {prompt}")

# #         response = client.chat.completions.create(
# #             model='gpt-4o',
# #             messages=[{"role": "user", "content": prompt}],
# #             temperature=0.7,
# #             response_format={
# #                 "type": "json_schema",
# #                 "json_schema": {
# #                     "name": "TriviaQuestion",
# #                     "schema": {
# #                         "type": "object",
# #                         "properties": {
# #                             "question": {"type": "string"},
# #                             "options": {
# #                                 "type": "object",
# #                                 "properties": {
# #                                     "A": {"type": "string"},
# #                                     "B": {"type": "string"},
# #                                     "C": {"type": "string"},
# #                                     "D": {"type": "string"}
# #                                 },
# #                                 "required": ["A", "B", "C", "D"]
# #                             },
# #                             "answer": {"type": "string"},
# #                             "difficulty": {"type": "string", "enum": ["easy", "medium", "hard", "expert"]},
# #                             "topic": {
# #                                 "type": "string",
# #                                 "enum": [
# #                                     "nigerian_history",
# #                                     "nigerian_entertainment",
# #                                     # "nigerian_sports",
# #                                     "human_anatomy"
# #                                 ]
# #                             }
# #                         },
# #                     }
# #                 }
# #             }
# #         )

# #         time.sleep(1)  # To avoid hitting rate limits too quickly
# #         response = json.loads(response.choices[0].message.content)
# #         print(f"✅ Generated {response} question(s) for {context}.")
# #         print(f" Data Type of response: {type(response)}")
# #         return response
# #     except Exception as e:
# #         print(f"API Error: {e}")
# #         return []


# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
# def generate_questions(folder_path, existing_questions, num_questions):
#     """Generate validated trivia questions using the provided schema."""
#     context = " > ".join(folder_path.split(os.sep)[-2:])
#     if existing_questions:
#         existing_qs = json.dumps(existing_questions, indent=2, ensure_ascii=False)
#     else:
#         existing_qs = "None yet."

#     try:
#         prompt = f"""
#         You are an expert Nigerian trivia quizmaster with deep, niche knowledge of **{context}** — its history, culture, people, and details.

#         🎯 **Your Task:**
#         Generate **exactly {num_questions} multiple-choice trivia questions** (MCQs) on **{context}**. Each question must be:
#         - Completely **unique** (no overlap in concept, phrasing, or fact with any other question — in this batch or below).
#         - Fun, insightful, verifiable, and not easily Googleable.
#         - Based on **distinct** knowledge points — no repetition or rephrasing of ideas.

#         🚫 **Avoid Duplicates**:
#         Do NOT copy, reword, or create semantically similar questions to:
#         {existing_qs if existing_qs else "None yet."}

#         📄 **Format (JSON ONLY — No extras):**
#         Return output as a **JSON object** with a "questions" array like:
#         {{
#         "questions": [
#             {{
#             "question": "...",
#             "options": {{
#                 "A": "...", "B": "...", "C": "...", "D": "..."
#             }},
#             "answer": "...",
#             "difficulty": "...",
#             "topic": "..."
#             }}
#         ]
#         }}

#         ⚠️ **Strict Rules**:
#         - No markdown, no headers, no explanations.
#         - 4 options (A-D) per question. Only 1 correct answer.
#         - Difficulty must be evenly distributed: `easy`, `medium`, `hard`, `expert`.
#         - Use varied formats (not just "What is...", "Who..."). Include fill-in-the-blanks, scenarios, etc.
#         - Keep language concise. Avoid long questions/options.
#         - Use consistent terms (e.g., always "Igbo", not "Ibo").
#         - Integrate Nigerian English or cultural references where it makes sense.
#         - Avoid jargon unless the topic truly requires it.

#         ✅ Output ONLY the JSON array. Nothing else.
#         """
#         print(f"📝 Full Prompt: {prompt}")

#         response = client.chat.completions.create(
#             model='gpt-4o',
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.7,
#             response_format={
#                 "type": "json_schema",
#                 "json_schema": {
#                     "name": "TriviaQuestionsWrapper",
#                     "schema": {
#                         "type": "object",  # Must be object for OpenAI API
#                         "properties": {
#                             "questions": {  # Wrap the array in an object property
#                                 "type": "array",
#                                 "items": {
#                                     "type": "object",
#                                     "properties": {
#                                         "question": {"type": "string"},
#                                         "options": {
#                                             "type": "object",
#                                             "properties": {
#                                                 "A": {"type": "string"},
#                                                 "B": {"type": "string"},
#                                                 "C": {"type": "string"},
#                                                 "D": {"type": "string"}
#                                             },
#                                             "required": ["A", "B", "C", "D"]
#                                         },
#                                         "answer": {"type": "string"},
#                                         "difficulty": {"type": "string", "enum": ["easy", "medium", "hard", "expert"]},
#                                         "topic": {
#                                             "type": "string",
#                                             "enum": [
#                                                 "nigerian_history",
#                                                 "nigerian_entertainment",
#                                                 "human_anatomy"
#                                             ]
#                                         }
#                                     },
#                                     "required": ["question", "options", "answer", "difficulty", "topic"]
#                                 },
#                                 "minItems": num_questions,
#                                 "maxItems": num_questions
#                             }
#                         },
#                         "required": ["questions"]
#                     }
#                 }
#             }
#         )

#         time.sleep(1)  # To avoid hitting rate limits too quickly
#         response_content = response.choices[0].message.content
#         response_data = json.loads(response_content)
        
#         # Extract the questions array from the wrapper object
#         questions_list = response_data.get("questions", [])
        
#         # Add debugging to see what we got
#         print(f"✅ Generated {len(questions_list)} question(s) for {context}.")
#         print(f"📊 Data Type of response: {type(questions_list)}")
#         print(f"🔍 First question preview: {questions_list[0]['question'][:50]}..." if questions_list else "No questions generated")
        
#         return questions_list
        
#     except Exception as e:
#         print(f"❌ API Error: {e}")
#         print(f"🔍 Response content (first 200 chars): {response.choices[0].message.content[:200] if 'response' in locals() else 'No response'}")
#         return []

# def main():
#     TOTAL_TO_GENERATE = 1
#     CHUNK_SIZE = 1
#     SIMILARITY_THRESHOLD = 0.5
#     existing_questions = []


#     folders = [
#         'nigerian_history',
#         'nigerian_entertainment',
#         # 'nigerian_sports',
#         'human_anatomy'
#     ]

#     base_dir = os.getcwd()

#     for topic_folder_name in folders:
#         topic_path = os.path.join(base_dir, topic_folder_name)
#         if not os.path.exists(topic_path):
#             print(f"⚠️ Folder {topic_path} does not exist. Skipping.")
#             continue
#     # '/human_anatomy'
#     # Walk through each folder in the current working directory
#     # for foldername, _, filenames in os.walk(os.getcwd()):
#         for foldername, _, filenames in os.walk(topic_path):
#             json_path = os.path.join(foldername, "previous_questions.json")
            
#             # If the file doesn't exist, create it with an empty list.
#             if "previous_questions.json" not in filenames:
#                 with open(json_path, "w") as f:
#                     json.dump([], f)

#             # Now read the file safely
#             try:
#                 with open(json_path, "r") as f:
#                     file_content = f.read().strip()
#                     if file_content:
#                         # load previous questions from current path
#                         existing = json.loads(file_content)    
#                         existing_questions = existing 
#                         print(f"📂 Found {len(existing_questions)} existing questions in {json_path}")               
#                     else:
#                         existing = []
#             except json.JSONDecodeError:
#                 existing = []

#             existing_count = len(existing)
#             print(f"🔍 Current path for previous questions contains {existing_count} questions")
#             if existing_count >= TOTAL_TO_GENERATE:
#                 print(f"✔️ {json_path} already has {existing_count} questions. Skipping.")
#                 continue

#             questions_to_generate = TOTAL_TO_GENERATE - existing_count
#             print(f"🧠 {json_path} has {existing_count} questions. Generating {questions_to_generate} more.")

#             while questions_to_generate > 0:
#                 batch_size = min(CHUNK_SIZE, questions_to_generate)
                
#                 new_q = generate_questions(foldername, existing_questions, num_questions=batch_size)
#                 if not new_q:
#                     print(f"⚠️ Failed to generate batch of {batch_size} questions.")
#                     continue

#                 # Deduplicate based on question text (case-insensitive)
#                 existing_texts = {q["question"].lower().strip() for q in existing}
#                 unique_new = [q for q in new_q if q["question"].lower().strip() not in existing_texts]
#                 print(f"📝 Generated {len(unique_new)} unique new question(s) in this batch.")

#                 if unique_new:
#                     # Path to centralized question store
#                     central_json = os.path.join(os.path.dirname(__file__), "all_questions.json")
#                     central_csv = os.path.join(os.path.dirname(__file__), "all_questions.csv")

#                     # Load existing centralized data
#                     if os.path.exists(central_json):
#                         with open(central_json, "r", encoding="utf-8") as f:
#                             try:
#                                 central_data = json.load(f)
#                             except json.JSONDecodeError:
#                                 central_data = []
#                     else:
#                         central_data = []

#                     # check if simiar vector already exists in ChromaDB
#                     # e
#                     # sim_score = is_similar()

#                     # Avoid duplicates in the central store too
#                     central_questions_text = {q["question"].lower().strip() for q in central_data}
#                     central_new_questions = [q for q in unique_new if q["question"].lower().strip() not in central_questions_text]

#                     if central_new_questions:
#                         # Update central JSON file
#                         with open(central_json, "w", encoding="utf-8") as f:
#                             json.dump(central_data + central_new_questions, f, indent=2, ensure_ascii=False)

#                         # Also write to central CSV
#                         csv_exists = os.path.exists(central_csv)
#                         with open(central_csv, "a", newline='', encoding='utf-8') as csvfile:
#                             writer = csv.DictWriter(csvfile, fieldnames=["question", "difficulty", "answer", "options 1", "options 2", "options 3", "options 4", "topic"])
#                             if not csv_exists:
#                                 writer.writeheader()
#                             for q in central_new_questions:
#                                 options = q.get("options", {})
#                                 row = {
#                                     "question": q["question"],
#                                     "difficulty": q["difficulty"],
#                                     "answer": q["answer"],
#                                     "options 1": options.get("A", {}),
#                                     "options 2": options.get("B", {}),
#                                     "options 3": options.get("C", {}),
#                                     "options 4": options.get("D", {}),
#                                     "topic": q["topic"]
#                                 }
#                                 writer.writerow(row)

#                         print(f"📦 Added {len(central_new_questions)} new question(s) to central store.")

#                     combined_questions = existing + unique_new
#                     with open(json_path, "w") as f:
#                         json.dump(combined_questions, f, indent=2)
#                     print(f"✅ Added question(s) to {json_path}")

#                     for question in unique_new:
#                         q_embedding = get_embedding(question['question'])
#                         print(f"🔍 Checking similarity for question: {question['question']}")
#                         sim_score= is_similar(embedding=q_embedding, collection=QUESTION_STORE, similarity_threshold=SIMILARITY_THRESHOLD)
#                         if sim_score:
#                             print(f"⚠️ Skipping: Similar question already exists in Chroma: {question['question']}")
#                             continue
#                         print(f"✅ No similar question found in Chroma for: {question['question']}")
#                         # If the question is similar, skip adding it to ChromaDB
#                         metadata = [{'source': 'folders', 'timestamp': time.time()}]
#                         success = add_to_chroma(
#                             collection=QUESTION_STORE,
#                             docs=[question['question']],
#                             embeddings=[q_embedding],
#                             ids=[generate_hash(question['question'])],
#                             metadata=metadata
#                         )
#                         # print(f"✅ Added question to ChromaDB: {question['question']}")
#                         # if success:
#                         #     print(f"✅ Added question to ChromaDB: {question['question']}")

#                     questions_to_generate = 1 - len(existing)


# if __name__ == "__main__":
#     main()

import os
import time
import json
import csv
import hashlib
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import threading
from queue import Queue
import random

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

chroma_client = chromadb.PersistentClient(path="chroma")
QUESTION_STORE = chroma_client.get_or_create_collection(name="quizme_test_1")

# Thread-safe locks for file operations
file_locks = defaultdict(threading.Lock)
central_lock = threading.Lock()

# Similarity checking parameters
SIMILARITY_THRESHOLD = 0.3

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

def is_similar(embedding, collection, similarity_threshold):
    """
    Checks if the given embedding is semantically similar to any existing item in the collection.
    """
    if embedding is None:
        return False

    embedding_values = embedding[0] if isinstance(embedding, list) and isinstance(embedding[0], list) else embedding
    current_collection_size = collection.count()
    n_results_to_query = min(current_collection_size, 50000)
    
    if n_results_to_query == 0:
        return False

    results = collection.query(
        query_embeddings=[embedding_values],
        n_results=n_results_to_query,
        include=['distances', 'documents']
    )

    if not results or not results["distances"] or not results["distances"][0]:
        return False

    for dist in results["distances"][0]:
        if dist < (1 - similarity_threshold):
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
        print(f"✅ Successfully added {len(docs)} documents to ChromaDB.")
        return True
    except Exception as e:
        print(f"❌ Failed to add to ChromaDB: {e}")
        return False

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=3))
def generate_questions_batch(topic_info, num_questions):
    """Generate questions for a specific topic/subfolder"""
    folder_path, existing_questions = topic_info
    context = " > ".join(folder_path.split(os.sep)[-2:])
    
    # Sample existing questions to avoid repetition
    if existing_questions and len(existing_questions) > 5:
        # sample_existing = random.sample(existing_questions, 5)
        # existing_qs = f"Avoid similar to: {[q['question'][:50] + '...' for q in sample_existing]}"
        existing_qs = f"Avoid similar to: {[q['question'] + '...' for q in existing_questions]}"
    elif existing_questions:
        existing_qs = f"Avoid similar to: {[q['question'] + '...' for q in existing_questions]}"
    else:
        existing_qs = "None yet."

    try:
        topic_name = context.split('/')[-1] if '/' in context else context.split('\\')[-1]
        
        prompt = f"""
        You are an expert Nigerian trivia quizmaster with deep, niche knowledge of **{context}** — its history, culture, people, and details.

        🎯 **Your Task:**
        Generate **exactly {num_questions} multiple-choice trivia questions** (MCQs) on **{context}**. Each question must be:
        - Completely **unique** (no overlap in concept, phrasing, or fact with any other question — in this batch or below).
        - Fun, insightful, verifiable, and not easily Googleable.
        - Based on **distinct** knowledge points — no repetition or rephrasing of ideas.

        🚫 **Avoid Duplicates**:
        Do NOT copy, reword, or create semantically similar questions to:
        {existing_qs if existing_qs else "None yet."}

        📄 **Format (JSON ONLY — No extras):**
        Return output as a **JSON array** like:
                JSON format:
                {{
                "questions": [
                    {{
                    "question": "...",
                    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
                    "answer": "A",
                    "difficulty": "easy",
                    "topic": "{topic_name}"
                    }}
                ]
                }}

        ⚠️ **Strict Rules**:
        - No markdown, no headers, no explanations.
        - 4 options (A–D) per question. Only 1 correct answer.
        - Difficulty must be evenly distributed: `easy`, `medium`, `hard`, `expert`.
        - Use varied formats (not just “What is...”, “Who...”). Include fill-in-the-blanks, scenarios, etc.
        - Keep language concise. Avoid long questions/options.
        - Use consistent terms (e.g., always “Igbo”, not “Ibo”).
        - Integrate Nigerian English or cultural references where it makes sense.
        - Avoid jargon unless the topic truly requires it.

        ✅ Output ONLY the JSON array. Nothing else.
        """

        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            # max_tokens=1500,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "TriviaQuestionsWrapper",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "questions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "question": {"type": "string"},
                                        "options": {
                                            "type": "object",
                                            "properties": {
                                                "A": {"type": "string"},
                                                "B": {"type": "string"},
                                                "C": {"type": "string"},
                                                "D": {"type": "string"}
                                            },
                                            "required": ["A", "B", "C", "D"]
                                        },
                                        "answer": {"type": "string"},
                                        "difficulty": {"type": "string", "enum": ["easy", "medium", "hard", "expert"]},
                                        "topic": {
                                            "type": "string",
                                            "enum":[
                                                "nigerian_history",
                                                "nigerian_entertainment",
                                                # "nigerian_sports",
                                                "human_anatomy"
                                            ]}
                                    },
                                    "required": ["question", "options", "answer", "difficulty", "topic"]
                                },
                                # "minItems": num_questions,
                                # "maxItems": num_questions
                            }
                        },
                        "required": ["questions"]
                    }
                }
            }
        )

        response_content = response.choices[0].message.content
        response_data = json.loads(response_content)
        questions_list = response_data.get("questions", [])
        
        # Filter out similar questions using embeddings
        filtered_questions = []
        for question in questions_list:
            print(f"🔍 Checking similarity for: {question['question'][:50]}...")
            
            # Get embedding for the question
            q_embedding = get_embedding(question['question'])
            if q_embedding is None:
                print(f"⚠️ Skipping question due to embedding failure: {question['question']}")
                continue
            
            # Check if similar question exists in ChromaDB
            if is_similar(embedding=q_embedding, collection=QUESTION_STORE, similarity_threshold=SIMILARITY_THRESHOLD):
                print(f"⚠️ Skipping similar question: {question['question'][:50]}...")
                continue
            
            # Check against existing questions in current batch (text-based for now)
            existing_texts = {q['question'].lower().strip() for q in filtered_questions}
            if question['question'].lower().strip() in existing_texts:
                print(f"⚠️ Skipping duplicate in batch: {question['question'][:50]}...")
                continue
            
            # Question passed all similarity checks
            filtered_questions.append(question)
            print(f"✅ Question approved: {question['question'][:50]}...")
        
        print(f"✅ Generated {len(questions_list)} questions, {len(filtered_questions)} passed similarity check for {context}")
        return folder_path, filtered_questions
        
    except Exception as e:
        print(f"❌ API Error for {context}: {e}")
        return folder_path, []

def save_questions_safely(folder_path, new_questions, all_existing_questions):
    """Thread-safe function to save questions to files and ChromaDB"""
    json_path = os.path.join(folder_path, "previous_questions.json")
    
    with file_locks[json_path]:
        # Re-read the file to get latest state
        try:
            with open(json_path, "r") as f:
                current_questions = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            current_questions = []
        
        # Deduplicate using text comparison
        existing_texts = {q["question"].lower().strip() for q in current_questions}
        unique_new = [q for q in new_questions if q["question"].lower().strip() not in existing_texts]
        
        if unique_new:
            # Add to ChromaDB with similarity checking
            chroma_approved_questions = []
            for question in unique_new:
                q_embedding = get_embedding(question['question'])
                if q_embedding is None:
                    print(f"⚠️ Skipping ChromaDB add due to embedding failure: {question['question']}")
                    continue
                
                # Final similarity check before adding to ChromaDB
                if not is_similar(embedding=q_embedding, collection=QUESTION_STORE, similarity_threshold=SIMILARITY_THRESHOLD):
                    metadata = {'source': 'folders', 'timestamp': time.time()}
                    success = add_to_chroma(
                        collection=QUESTION_STORE,
                        docs=[question['question']],
                        embeddings=[q_embedding],
                        ids=[generate_hash(question['question'])],
                        metadata=[metadata]
                    )
                    if success:
                        chroma_approved_questions.append(question)
                        print(f"✅ Added to ChromaDB: {question['question'][:50]}...")
                    else:
                        print(f"❌ Failed to add to ChromaDB: {question['question'][:50]}...")
                else:
                    print(f"⚠️ Skipping ChromaDB add (similar found): {question['question'][:50]}...")
            
            # Update local file with all unique questions (even if some weren't added to ChromaDB)
            updated_questions = current_questions + unique_new
            with open(json_path, "w") as f:
                json.dump(updated_questions, f, indent=2)
            
            # Update central files
            with central_lock:
                update_central_files(unique_new)
            
            print(f"💾 Saved {len(unique_new)} questions to {folder_path} ({len(chroma_approved_questions)} added to ChromaDB)")
            return unique_new
    
    return []

def update_central_files(new_questions):
    """Update centralized question storage"""
    central_json = os.path.join(os.path.dirname(__file__), "all_questions.json")
    central_csv = os.path.join(os.path.dirname(__file__), "all_questions.csv")
    
    # Update JSON
    try:
        with open(central_json, "r", encoding="utf-8") as f:
            central_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        central_data = []
    
    central_questions_text = {q["question"].lower().strip() for q in central_data}
    central_new_questions = [q for q in new_questions if q["question"].lower().strip() not in central_questions_text]
    
    if central_new_questions:
        with open(central_json, "w", encoding="utf-8") as f:
            json.dump(central_data + central_new_questions, f, indent=2, ensure_ascii=False)
        
        # Update CSV
        csv_exists = os.path.exists(central_csv)
        with open(central_csv, "a", newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["question", "difficulty", "answer", "options 1", "options 2", "options 3", "options 4", "topic"])
            if not csv_exists:
                writer.writeheader()
            for q in central_new_questions:
                options = q.get("options", {})
                row = {
                    "question": q["question"],
                    "difficulty": q["difficulty"],
                    "answer": q["answer"],
                    "options 1": options.get("A", ""),
                    "options 2": options.get("B", ""),
                    "options 3": options.get("C", ""),
                    "options 4": options.get("D", ""),
                    "topic": q["topic"]
                }
                writer.writerow(row)

def get_all_topic_folders():
    """Get all topic folders and their current question counts"""
    folders = [
        'nigerian_history',
        'nigerian_entertainment',
        'human_anatomy'
    ]
    
    base_dir = os.getcwd()
    topic_info = []
    
    for topic_folder_name in folders:
        topic_path = os.path.join(base_dir, topic_folder_name)
        if not os.path.exists(topic_path):
            print(f"⚠️ Folder {topic_path} does not exist. Skipping.")
            continue
            
        for foldername, _, filenames in os.walk(topic_path):
            json_path = os.path.join(foldername, "previous_questions.json")
            
            if "previous_questions.json" not in filenames:
                with open(json_path, "w") as f:
                    json.dump([], f)
                existing_questions = []
            else:
                try:
                    with open(json_path, "r") as f:
                        file_content = f.read().strip()
                        existing_questions = json.loads(file_content) if file_content else []
                except json.JSONDecodeError:
                    existing_questions = []
            
            topic_info.append({
                'folder_path': foldername,
                'existing_count': len(existing_questions),
                'existing_questions': existing_questions,
                'topic_name': " > ".join(foldername.split(os.sep)[-2:])
            })
    
    return topic_info

def main():
    MAX_QUESTIONS_PER_TOPIC = 500  # Target per topic
    QUESTIONS_PER_BATCH = 50       # Generate 50 questions per API call
    MAX_CONCURRENT_REQUESTS = 3   # Parallel API calls
    TOTAL_GENERATION_ROUNDS = 20  # How many rounds to run
    
    print("🚀 Starting balanced multi-topic question generation with similarity detection...")
    print(f"📊 Similarity threshold: {SIMILARITY_THRESHOLD} (higher = more strict)")
    
    # Get all topics and their current status
    all_topics = get_all_topic_folders()
    
    print(f"📊 Found {len(all_topics)} topic folders:")
    for topic in all_topics:
        print(f"  - {topic['topic_name']}: {topic['existing_count']} questions")
    
    # Round-robin generation
    for round_num in range(1, TOTAL_GENERATION_ROUNDS + 1):
        print(f"\n🔄 === ROUND {round_num}/{TOTAL_GENERATION_ROUNDS} ===")
        
        # Prepare batch jobs for this round
        batch_jobs = []
        for topic in all_topics:
            if topic['existing_count'] < MAX_QUESTIONS_PER_TOPIC:
                questions_needed = min(QUESTIONS_PER_BATCH, MAX_QUESTIONS_PER_TOPIC - topic['existing_count'])
                if questions_needed > 0:
                    batch_jobs.append((
                        (topic['folder_path'], topic['existing_questions']),
                        questions_needed
                    ))
        
        if not batch_jobs:
            print("✅ All topics have reached their target question count!")
            break
        
        print(f"📝 Processing {len(batch_jobs)} topics in parallel...")
        
        # Execute batches in parallel
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
            future_to_topic = {
                executor.submit(generate_questions_batch, topic_info, num_questions): topic_info[0]
                for topic_info, num_questions in batch_jobs
            }
            
            for future in as_completed(future_to_topic):
                folder_path = future_to_topic[future]
                try:
                    folder_path, new_questions = future.result()
                    if new_questions:
                        # Find and update the topic info
                        for topic in all_topics:
                            if topic['folder_path'] == folder_path:
                                saved_questions = save_questions_safely(
                                    folder_path, 
                                    new_questions, 
                                    topic['existing_questions']
                                )
                                topic['existing_count'] += len(saved_questions)
                                topic['existing_questions'].extend(saved_questions)
                                break
                        
                except Exception as exc:
                    print(f"❌ Error processing {folder_path}: {exc}")
        
        # Show progress after each round
        print(f"\n📈 Progress after round {round_num}:")
        total_questions = 0
        for topic in all_topics:
            print(f"  - {topic['topic_name']}: {topic['existing_count']}/{MAX_QUESTIONS_PER_TOPIC}")
            total_questions += topic['existing_count']
        print(f"📊 Total questions generated: {total_questions}")
        
        # Small delay between rounds
        time.sleep(1)
    
    print("\n🎉 Generation complete!")
    final_total = sum(topic['existing_count'] for topic in all_topics)
    print(f"📊 Final count: {final_total} questions across all topics")
    print(f"🔍 ChromaDB collection size: {QUESTION_STORE.count()}")

if __name__ == "__main__":
    main()