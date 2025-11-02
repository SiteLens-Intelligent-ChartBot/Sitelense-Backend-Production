import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import time, random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient

# -----------------------------
# MongoDB Atlas Setup
# -----------------------------
MONGO_URI = "mongodb+srv://sunilsahoo:2664@cluster0.yp0utdu.mongodb.net"
DB_NAME = "sitelense_chats"
COLLECTION_NAME = "sitelense_ai"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# -----------------------------
# Gemini Setup (Optional)
# -----------------------------
import google.generativeai as genai
genai.configure(api_key="AIzaSyAPmQlWLJz3XH2PcuoGujeN1okniir6DTU")

USE_GEMINI = False  # ✅ Keep False for best speed & low memory
gemini_model = None
if USE_GEMINI:
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# -----------------------------
# Global Model + Cache
# -----------------------------
_model = None
_cached_docs = []


def get_model():
    """Load the SentenceTransformer model only once."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print("🔹 Loading embedding model into memory...")
        _model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    return _model


def refresh_cache():
    """Load all knowledge base entries from MongoDB into memory."""
    global _cached_docs
    _cached_docs = list(collection.find({}, {"_id": 1, "text": 1, "embedding": 1}))
    print(f"✅ Cache loaded with {len(_cached_docs)} entries.")


# Preload model + cache on startup
get_model()
refresh_cache()

# -----------------------------
# Helper: Gemini Rewriter
# -----------------------------
def safe_rewrite(original_text: str) -> str:
    """Use Gemini to rewrite the response naturally, or fallback."""
    if not USE_GEMINI:
        return original_text.split("Context: ", 1)[1]

    for attempt in range(3):
        try:
            response = gemini_model.generate_content(
                f"Please answer the user’s question naturally using this information:\n\n{original_text}"
            )
            if response and response.text:
                return response.text
            else:
                return original_text.split("Context: ", 1)[1]
        except Exception as e:
            err = str(e)
            if "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                print("❌ Gemini quota exceeded → fallback.")
                return original_text.split("Context: ", 1)[1]
            print(f"⚠️ Gemini error on attempt {attempt+1}: {err}")
            time.sleep(1 + random.random())

    return original_text.split("Context: ", 1)[1]


# -----------------------------
# Core Functions
# -----------------------------
def answer_question(query: str) -> str:
    """Return the most relevant stored text for the given query."""
    global _cached_docs

    if not _cached_docs:
        refresh_cache()
    if not _cached_docs:
        return "No knowledge available yet. Please add statements first."

    model = get_model()
    query_embedding = model.encode([query]).reshape(1, -1)

    best_doc = None
    best_score = -1.0

    for doc in _cached_docs:
        if "embedding" not in doc:
            continue
        emb = np.array(doc["embedding"]).reshape(1, -1)
        sim = cosine_similarity(query_embedding, emb)[0][0]
        if sim > best_score:
            best_score = sim
            best_doc = doc

    if best_doc and best_score >= 0.4:
        return safe_rewrite(f"Q: {query}\nContext: {best_doc['text']}")

    return "I don't know the answer to that yet."


def add_statement(text: str):
    """Insert a new text statement + embedding into MongoDB and memory cache."""
    global _cached_docs
    model = get_model()
    emb = model.encode([text])[0].tolist()
    collection.insert_one({"text": text, "embedding": emb})
    _cached_docs.append({"text": text, "embedding": emb})
    print(f"✅ Added statement: {text[:50]}...")


def delete_statement(text: str):
    """Delete the latest matching statement."""
    global _cached_docs
    doc = collection.find_one({"text": text}, sort=[("_id", -1)])
    if doc:
        collection.delete_one({"_id": doc["_id"]})
        _cached_docs = [d for d in _cached_docs if d.get("text") != text]
        print(f"🗑️ Deleted statement: {text[:50]}...")
        return True
    return False


def list_statements():
    """List all stored statements (latest → oldest)."""
    global _cached_docs
    if not _cached_docs:
        refresh_cache()
    return [doc["text"] for doc in reversed(_cached_docs)]
