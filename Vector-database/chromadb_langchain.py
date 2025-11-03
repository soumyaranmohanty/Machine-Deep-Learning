#Here we will be using the ChromaDB vector database to store and retrieve embeddings.
"""
Steps to store the data to vector database:
1. Install ChromaDB library if not already installed.
2. Import necessary modules from ChromaDB.
3. Initialize a ChromaDB client.
4. Create a collection in the ChromaDB database.
5. Read the data from the source (e.g., CSV file).
6. Generate embeddings for the data using a suitable embedding model.
7. Embedding model could be OpenAI, HuggingFace, Llama2, google palm embedding model etc.
8. Insert the embeddings along with their corresponding metadata into the ChromaDB collection.
"""

#Here in this code base, we will use langchain and openai embedding model to store the data to chromadb vector database.
#Reference docs: https://docs.langchain.com/oss/python/integrations/vectorstores
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




embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Access the underlying OpenAI client
try:
    # Get the API key from the client
    client_api_key = embeddings.client.api_key
    print(f"🔑 Client API Key: {client_api_key}")
except AttributeError:
    print("⚠️ Unable to access client API key directly")




#Now read the data from the source, generation of embeddings for the data is handeled by vector_store.

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader

loader = DirectoryLoader("./Vector-database/new_articles", glob = "./*.txt", loader_cls= TextLoader)
document = loader.load()


from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
text = text_splitter.split_documents(document)


#vector_store = Chroma(
#collection_name="Collection1",
#embedding_function=embeddings,
#persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
#)

"""
vectordb = Chroma.from_documents(documents=text,
                                 embedding=embeddings,
                                 persist_directory="./chroma_langchain_db")

"""

"""

Method 1 : 
Chroma()
Creates an empty ChromaDB collection
You need to manually add documents later using add_documents()
More control over the process

Method 2 :
Chroma.from_documents()
Creates ChromaDB collection and adds documents in one step
Documents are automatically processed and added
More convenient for initial setup


Key Differences:
Aspect	            Chroma() Constructor	                   Chroma.from_documents()
Initial State	        Empty collection	                       Collection with documents
Document Addition	    Manual (add_documents())	               Automatic
Use Case	            When you want to add documents later	   When you have documents ready
Control	                More granular control	                   All-in-one convenience


"""


#vector_store.add_documents(documents=text, ids=[str(i) for i in range(len(text))])

#vector_store.persist()  # Save to disk if persist_directory is set

#print("Data has been added to ChromaDB vector database successfully.")





