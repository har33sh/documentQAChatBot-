
from module_rag.chatbot import ChatBot
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile


router = APIRouter()
bot = ChatBot()


class ChatRequest(BaseModel):
    user_id: str = "123"
    question: str


@router.post("/chat")
async def qa_chat(request: ChatRequest):
    response = bot.chat(request.question, request.user_id)
    return JSONResponse(response)


@router.post("/voice")
async def voice_chat(audio: UploadFile = File(...)):
    # Send file to Whisper API for transcription
    file_content = await audio.read()
    response = bot.voice_chat(file_content)
    return response
