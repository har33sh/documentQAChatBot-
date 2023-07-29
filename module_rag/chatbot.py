from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from module_rag.doc_loader import DocumentLoader
import io
from pydub import AudioSegment
from config import WHISPER_API_KEY, WHISPER_API
import requests

class ChatBot:
    def __init__(self):
        self.vectorstore = self.get_vectorstore()
        self.cache = {}

    @staticmethod
    def get_vectorstore():
        document = DocumentLoader()
        return document.vectorstore

    def chat(self, question, user_id="123"):
        if user_id not in self.cache:
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            retriever = self.vectorstore.as_retriever()
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            chats = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
            self.cache[user_id] = chats

        result = self.cache[user_id]({"question": question})
        return result['answer']

    def voice_chat(self, file_content):
        try:
            audio_file = io.BytesIO(file_content)
            audio_segment = AudioSegment.from_file(audio_file)
            audio_wav_file = io.BytesIO()
            audio_segment.export(audio_wav_file, format="wav")

            headers = {
                'Authorization': f'Bearer {WHISPER_API_KEY}'
            }

            file_data = {"file": ("audio.wav", audio_wav_file.getvalue(), "audio/wav")}
            data = {
                "fileType": "wav",
                "diarization": "false",
                "numSpeakers": "1",
                "initialPrompt": "",
                "language": "en",
                "task": "transcribe",
                "callbackURL": ""
            }

            response = requests.post(WHISPER_API, headers=headers, files=file_data, data=data)
            transcription = response.json()['text']
            print(transcription)
            return self.chat(transcription)
        except Exception as error:
            return {"text": "Error occurred", "error": error}
