import chromadb
import os
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

current_dir = os.getcwd()
data_folder_path = os.path.join(current_dir, "data")

client = chromadb.PersistentClient(path = data_folder_path)

openai_ef = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"),model_name="text-embedding-3-small")


client.create_collection(name="test-collection",embedding_function=openai_ef)

collection =  client.get_collection(name="test-collection")

collection.delete(["id1","id2","id3"])
collection.add(
    ids=["id1","id2","id3"], 
    documents=[
        "document about animals",
        "document about fruits",
        "document about cities"
    ])

results = collection.query(
    query_texts=["this is a query about madrid"], # Chroma will embed this for you
    n_results=1 # how many results to return
)
print(results)

#print(collection.get(["id1"]))
#print(collection.peek())
#print(collection.count())
