
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

embedding  = DefaultEmbeddingFunction()

#Load all files from the directory and read the data


import document_loader as dl
import text_chunker as tc

loader = dl.DocumentLoader()


directory_path = "./Vector-database/new_articles"

load_directory = loader.load_directory(directory_path)

#print(load_directory[:2])  # Print first two loaded documents for verification

print(len(load_directory))

#Chunk the loaded documents

all_chunks = []

for doc in load_directory:
    chunker = tc.TextChunker()
    chunks = chunker.chunk_document(doc, strategy="recursive")
    #print(f"Document: {doc['metadata']['filename']} - Chunks created: {len(chunks)}")
    all_chunks.extend(chunks)
print(f"Total chunks created from all documents: {len(all_chunks)}")

#Print first 2 chunks for verification
#for chunk in all_chunks[:2]:
    #print(chunk)



#Embedding the chunks and store in ChromaDB



client = chromadb.PersistentClient(path="./Vector-database")

try :
    collection = client.create_collection(
        name="my_collection",
        embedding_function=embedding
    )
    print("✅ Collection created successfully.")
except : 

    print("ℹ️ Collection already exists. Retrieved existing collection.")

collection.add(documents =[doc['content'] for doc in all_chunks], 
               metadatas=[doc['metadata'] for doc in all_chunks], 
               ids=[str(i) for i in range(len(all_chunks))]
               )



#Now the data is stored in ChromaDB with embeddings
print("✅ Data added to ChromaDB collection with embeddings.")





