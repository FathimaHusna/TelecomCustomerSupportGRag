import os
import sys
import shutil
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List

# Ensure sibling modules are importable when running as a package (uvicorn src.api:app)
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from chatbot import TelecomSupportChatbot


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    response: str
    model: Optional[str] = None


def create_app() -> FastAPI:
    app = FastAPI(title="Telecom Support Chatbot API", version="1.0.0")

    # Initialize chatbot on startup
    chatbot_holder = {"bot": None}
    
    # Determine data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")

    @app.on_event("startup")
    def _startup():
        bot = TelecomSupportChatbot(data_dir)
        ok = bot.initialize()
        if not ok:
            # We don't raise here to keep /health working; /chat will return 500
            print("Chatbot failed to initialize during startup.")
        chatbot_holder["bot"] = bot

    @app.get("/health")
    def health():
        bot = chatbot_holder.get("bot")
        initialized = bool(bot and bot.graph_rag)
        return {"status": "ok", "initialized": initialized}

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest):
        bot = chatbot_holder.get("bot")
        if not bot or not bot.graph_rag:
            raise HTTPException(status_code=503, detail="Chatbot not initialized")
        if not req.query or not req.query.strip():
            raise HTTPException(status_code=400, detail="Query must be non-empty")
        text = bot.generate_response(req.query.strip())
        return ChatResponse(response=text, model=os.getenv("LOCAL_MODEL_NAME", "unknown"))

    @app.post("/reload")
    def reload_data():
        bot = chatbot_holder.get("bot")
        if not bot:
            raise HTTPException(status_code=503, detail="Chatbot not created")
        ok = bot.reload_data()
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to reload data")
        return {"status": "success", "message": "Knowledge graph and vector index reloaded"}

    @app.post("/upload")
    async def upload_files(files: List[UploadFile] = File(...)):
        # Validate filenames
        allowed_files = ["support_tickets.csv", "technical_manuals.csv", "escalation_records.csv"]
        for file in files:
            if file.filename not in allowed_files:
                raise HTTPException(status_code=400, detail=f"Invalid file: {file.filename}. Only {allowed_files} are allowed.")
        
        # Save files
        for file in files:
            file_path = os.path.join(data_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        
        # Trigger reload automatically
        bot = chatbot_holder.get("bot")
        if bot:
            bot.reload_data()
            
        return {"status": "success", "uploaded": [f.filename for f in files], "message": "Files uploaded and re-indexed"}

    return app


# For `uvicorn src.api:app` usage
app = create_app()
