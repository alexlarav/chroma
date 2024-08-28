from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_openai import ChatOpenAI
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os
import requests


os.environ["OPENAI_API_KEY"] = "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"
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
    #turbo_llm = ChatOpenAI(temperature=0.5, model_name='gpt-3.5-turbo')
    turbo_llm = ChatOpenAI(temperature=0.5, model_name='gpt-4o-mini')
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(turbo_llm, prompt)
    qa_chain = create_retrieval_chain(retriever, question_answer_chain)

    #qa_chain = RetrievalQA.from_chain_type(llm=turbo_llm, chain_type="stuff",retriever=retriever, return_source_documents=True)
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
        
    llm_response = qa_chain.invoke({"input": query})  
    print("\nRESPUESTA:")
    print(llm_response['answer'])
    #for source in llm_response["source_documents"]:
    #   print(source.metadata['source'])
    doQuery(qa_chain)


os.system('cls')
DoRetriever()
