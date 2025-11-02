import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import HashingVectorizer
from pymongo import MongoClient
import gc

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
# Lightweight embedding model
# -----------------------------
_vectorizer = HashingVectorizer(
    n_features=256,       # Small memory footprint (adjust if needed)
    alternate_sign=False,
    norm="l2",
)
_vectorizer.fit(["init"])  # Initialize once


def embed_text(text: str):
    """Generate lightweight embedding using HashingVectorizer."""
    vec = _vectorizer.transform([text])
    return vec.toarray()[0].astype(np.float16)


# -----------------------------
# Core Functions
# -----------------------------
def answer_question(query: str) -> str:
    """Return best-matching text using lightweight embeddings."""
    if collection.estimated_document_count() == 0:
        return "No knowledge available yet. Please add statements first."

    docs = collection.find({}, {"_id": 0, "text": 1, "embedding": 1})
    query_emb = embed_text(query)

    best_doc, best_score = None, -1.0

    for doc in docs:
        emb = np.array(doc.get("embedding", []), dtype=np.float16)

        # ✅ Skip mismatched or invalid embeddings
        if emb.shape[0] != query_emb.shape[0]:
            continue

        if np.linalg.norm(query_emb) == 0 or np.linalg.norm(emb) == 0:
            continue

        sim = float(cosine_similarity([query_emb], [emb])[0][0])
        if sim > best_score:
            best_score, best_doc = sim, doc

    del query_emb
    gc.collect()

    if best_doc and best_score >= 0.3:
        return best_doc["text"]
    return "I don't know the answer to that yet."


def add_statement(text: str):
    """Insert a new statement and store its hashed embedding."""
    emb = embed_text(text).tolist()
    collection.insert_one({"text": text, "embedding": emb})
    gc.collect()
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
    """Return all stored statements in reverse order."""
    docs = collection.find({}, {"_id": 0, "text": 1}).sort([("_id", -1)])
    return [d["text"] for d in docs]
