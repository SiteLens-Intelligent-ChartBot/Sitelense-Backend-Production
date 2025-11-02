from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chatbot import answer_question, add_statement, delete_statement, list_statements
from fastapi.middleware.cors import CORSMiddleware

# ------------------------------------
# FastAPI App Setup
# ------------------------------------
app = FastAPI(title="SiteLens AI (Lite)", version="1.0")

# Allow all origins (for testing or frontend use)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------
# Models
# ------------------------------------
class Question(BaseModel):
    query: str

class Statement(BaseModel):
    text: str


# ------------------------------------
# Routes
# ------------------------------------

@app.get("/")
def home():
    return {"message": "✅ SiteLens AI backend is live (Lite version)"}


# --- User Asks ---
@app.post("/ask")
def ask_question(question: Question):
    try:
        answer = answer_question(question.query)
        return {"ok": True, "query": question.query, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Admin Adds ---
@app.post("/admin/add")
def admin_add(statement: Statement):
    try:
        add_statement(statement.text)
        data = list_statements()
        return {"ok": True, "message": "✅ Statement added", "count": len(data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Admin Deletes ---
@app.delete("/admin/delete")
def admin_delete(statement: Statement):
    try:
        success = delete_statement(statement.text)
        data = list_statements()
        if not success:
            raise HTTPException(status_code=404, detail="❌ Statement not found")
        return {"ok": True, "message": "🗑️ Statement deleted", "count": len(data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Admin Lists ---
@app.get("/admin/list")
def admin_list():
    try:
        data = list_statements()
        return {"ok": True, "count": len(data), "statements": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------
# Run command (for local testing)
# ------------------------------------
# uvicorn app:app --host 0.0.0.0 --port 8080 --reload
