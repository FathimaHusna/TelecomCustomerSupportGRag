import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

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

    @app.on_event("startup")
    def _startup():
        # Determine data directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(script_dir), "data")
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

    return app


# For `uvicorn src.api:app` usage
app = create_app()
