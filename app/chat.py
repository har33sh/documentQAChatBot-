from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from module_rag.chatbot import ChatBot

router = APIRouter()
bot = ChatBot()


class ChatRequest(BaseModel):
    user_id: str = "123"
    question: str


@router.post("/chat")
async def article_operations(request: ChatRequest):
    response = bot.chat(request.question, request.user_id)
    return JSONResponse(response)
