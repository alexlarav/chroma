from langchain_chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_text_splitters import CharacterTextSplitter
import os
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb.utils import embedding_functions

#embedding_function = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"),model_name="text-embedding-3-small")
embedding_function = SentenceTransformerEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")


def embeddingDoc():
    current_dir = os.getcwd()
    Chroma.reset_collection
    filepath = input("RUTA DEL ARCHIVO: ")
    loader = PyPDFLoader(filepath)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=2)
    documents = text_splitter.split_documents(documents)
    collect =Chroma.from_documents(documents, embedding_function, persist_directory="data2/docs_db")
    data()

def data():
    collec = Chroma(persist_directory="data2/docs_db", embedding_function=embedding_function)
    pregunta(collec)

def pregunta(collec):
    query = input("PREGUNTA:   ")
    if (query=='exit'):
        embeddingDoc()
    docs = collec.similarity_search(query)
    print(f"RESPUESTA:   {docs[0].page_content}")
    pregunta(collec)

data()



