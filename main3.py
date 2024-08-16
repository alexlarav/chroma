import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader, PyPDFLoader, JSONLoader

from langchain_chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_text_splitters import CharacterTextSplitter, TextSplitter
import os
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain.document_loaders.pdf import PyPDFDirectoryLoader

embedding_funct = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"),model_name="text-embedding-3-small")
embedding_model = SentenceTransformerEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
persist_directory="data3/docs_db"
#embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")



def embeddingDoc():
    #folderpath ="C:\\Users\\admninistrador\\Documents\\Chorma\\documentspdf"
    folderpath = input("RUTA DEL DIRECTORIO: ")
    document_loader = PyPDFDirectoryLoader(folderpath) 
    documents = document_loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(documents)
    bd=Chroma.from_documents(documents, embedding_model, persist_directory=persist_directory) 
    data()

def data():
    db3 = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    pregunta(db3)

def pregunta(db):
    query = input("PREGUNTA:   ")
    if (query=='upload'):
        embeddingDoc()
    docs = db.similarity_search(query)
    print(f"RESPUESTA:   {docs[0].page_content}")
    pregunta(db)

data()



