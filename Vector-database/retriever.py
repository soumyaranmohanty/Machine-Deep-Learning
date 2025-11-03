"""
The below script is to retrieve data from the ChromaDB vector database using Langchain and OpenAI embedding model.

"""
import os
from dotenv import load_dotenv
load_dotenv()

# 1. Check OpenAI API Key
api_key = os.getenv('OPENAI_API_KEY')
if api_key:
    print(f"✅ OpenAI API Key: {api_key}...")
else:
    print("❌ OpenAI API Key: Not found")



from langchain_openai import OpenAIEmbeddings

from langchain_chroma import Chroma



#Load the existing vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(
    collection_name="Collection1",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where the data is saved locally
)





retriever = vector_store.as_retriever()


#docs = retriever.get_relevant_documents("Sample query to test retriever functionality.")

docs = retriever.invoke("Sample query to test retriever functionality.")