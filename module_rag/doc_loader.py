from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from config import *


class DocumentLoader:
    def __init__(self):
        self.vectorstore = None
        self.data = self.load_files()
        self.text_splits = self.splitter(self.data)
        self.vectorstore = self.store(self.text_splits)

    @staticmethod
    def load_files():
        loader = TextLoader("knowledge/doc.txt")
        data = loader.load()
        return data

    @staticmethod
    def splitter(data):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(data)
        return all_splits

    @staticmethod
    def store(all_splits):
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
        return vectorstore

    def get_vectorstore(self):
        return self.vectorstore
