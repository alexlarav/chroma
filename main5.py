from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_openai import ChatOpenAI
import requests
from bs4 import BeautifulSoup
import os

persist_directory = 'data5/docs_db'
embedding = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def fileUpload_url():
    url = input("\nRUTA URL: ")
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text_data = soup.get_text()
    texts = text_splitter.split_text(text_data)
    doDBURL(texts)

def fileUpload_pdf():
    folderpath = input("\nRUTA DEL DIRECTORIO: ")
    loader = DirectoryLoader(folderpath, glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    texts = text_splitter.split_documents(documents)
    doDBPDF(texts)

def doDBPDF(texts):
    Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
    DoRetriever()

def doDBURL(texts):
    Chroma.from_texts(texts=texts, embedding=embedding, persist_directory=persist_directory)
    DoRetriever()

def deleteDB(vectorDB):
    vectorDB.delete_collection()
    vectorDB.persist()

def DoRetriever():
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    retriever.search_type
    turbo_llm = ChatOpenAI(temperature=0.5, model_name='gpt-3.5-turbo')
    qa_chain = RetrievalQA.from_chain_type(llm=turbo_llm, chain_type="stuff",retriever=retriever, return_source_documents=True)
    doQuery(qa_chain)

def doQuery(qa_chain):
    print("\nPREGUNTA:")
    query = input()

    if(query.upper()=="PDF"):
        fileUpload_pdf()
    elif(query.upper() == "URL"):
        fileUpload_url()
    elif(query.upper()=="EXIT"):
        exit()
        
    llm_response = qa_chain.invoke(query)  
    print("\nRESPUESTA:")
    print(llm_response['result'])
    #for source in llm_response["source_documents"]:
    #   print(source.metadata['source'])
    doQuery(qa_chain)


os.system('cls')
DoRetriever()