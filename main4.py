from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.chat_models import ChatOpenAI
import os


persist_directory = 'data4/docs_db'
embedding = OpenAIEmbeddings()

def fileUpload():
    folderpath = input("\nRUTA DEL DIRECTORIO: ")
    loader = DirectoryLoader(folderpath, glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)
    doDB(texts)

def doDB(texts):
    vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
    DoRetriever()

def deleteDB(vectorDB):
    vectorDB.delete_collection()
    vectorDB.persist()

def DoRetriever():
    print("\nPREGUNTA:")
    query = input()
    if(query=="_"):
        fileUpload()
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    #retriever = vectordb.as_retriever()
    #docs = retriever.get_relevant_documents(query)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    retriever.search_type
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff",retriever=retriever, return_source_documents=True)
    llm_response = qa_chain(query)  
    process_llm_response(llm_response)
    DoRetriever()

def process_llm_response(llm_response):
    print("\nRESPUESTA:")
    print(llm_response['result'])
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

DoRetriever()