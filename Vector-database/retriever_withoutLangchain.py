#Connect to the persistent ChromaDB and retrieve data based on user query

import chromadb

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

embedding  = DefaultEmbeddingFunction()

vector_store = chromadb.PersistentClient(path="./Vector-database")
collection = vector_store.get_collection(
    name="my_collection",
    embedding_function=embedding
)

#Retrieve the data based on the user query
def retrieve_data(query):
    results = collection.query(
        query_embeddings=embedding.embed_query(query),
        n_results=5
    )
    return results


#Example query
query = "What is the impact of climate change on agriculture?"
results = retrieve_data(query)

print("Top 5 relevant chunks for the query:")
for i in range(len(results['documents'][0])):
    print(f"Chunk ID: {results['ids'][0][i]}")
    print(f"Content: {results['documents'][0][i]}")
    print(f"Metadata: {results['metadatas'][0][i]}")
    print("-----")