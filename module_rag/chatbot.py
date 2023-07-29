from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from module_rag.doc_loader import DocumentLoader


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
