import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
import time, gc

# -----------------------------
# MongoDB Atlas Setup
# -----------------------------
MONGO_URI = "mongodb+srv://sunilsahoo:2664@cluster0.yp0utdu.mongodb.net"
DB_NAME = "sitelense_chats"
COLLECTION_NAME = "sitelense_ai"

client = MongoClient(MONGO_URI, connect=False)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# -----------------------------
# Lightweight model loader
# -----------------------------
def get_temp_model():
    """Load MiniLM only when needed, free after use."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")

# -----------------------------
# Core Functions (Low RAM)
# -----------------------------
def answer_question(query: str) -> str:
    """Compute embedding, stream through Mongo docs, free model."""
    docs = collection.find({}, {"_id": 0, "text": 1, "embedding": 1})
    if collection.count_documents({}) == 0:
        return "No knowledge available yet. Please add statements first."

    # Load model only here
    model = get_temp_model()
    query_emb = model.encode([query])[0].astype(np.float16)
    del model
    gc.collect()

    best_doc, best_score = None, -1.0
    for doc in docs:
        emb = np.array(doc["embedding"], dtype=np.float16)
        sim = float(cosine_similarity([query_emb], [emb])[0][0])
        if sim > best_score:
            best_score, best_doc = sim, doc

    del query_emb
    gc.collect()

    if best_doc and best_score >= 0.4:
        return best_doc["text"]
    return "I don't know the answer to that yet."


def add_statement(text: str):
    """Encode and insert one new statement (auto memory cleanup)."""
    model = get_temp_model()
    emb = model.encode([text])[0].astype(np.float16).tolist()
    del model
    gc.collect()

    collection.insert_one({"text": text, "embedding": emb})
    print("✅ Added new statement.")


def delete_statement(text: str):
    """Delete the latest matching statement."""
    doc = collection.find_one({"text": text}, sort=[("_id", -1)])
    if doc:
        collection.delete_one({"_id": doc["_id"]})
        print("🗑️ Deleted statement.")
        return True
    return False


def list_statements():
    """Return all stored texts (no embeddings)."""
    docs = collection.find({}, {"_id": 0, "text": 1}).sort([("_id", -1)])
    return [d["text"] for d in docs]
