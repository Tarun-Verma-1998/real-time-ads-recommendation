from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import pickle
import numpy as np

# Connect to Milvus server
connections.connect("default", host="localhost", port="19530")
print(" Connected to Milvus.")

# Define collection schema
fields = [
    FieldSchema(name="ad_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)  # Dimension of T5 embedding
]
schema = CollectionSchema(fields, description="Ad Embeddings Collection")

# Create or load collection
collection_name = "ads_collection"
if not Collection.exists(collection_name):
    collection = Collection(name=collection_name, schema=schema)
    print(f" Collection '{collection_name}' created.")
else:
    collection = Collection(collection_name)
    print(f" Collection '{collection_name}' already exists.")

# Load embeddings
with open('embeddings/ads_embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

# Prepare data for insertion
ad_ids = list(range(1, len(embeddings) + 1))
data = [ad_ids, embeddings]

# Insert data into Milvus
if collection.num_entities == 0:
    collection.insert(data)
    print(f" Inserted {len(embeddings)} embeddings into Milvus.")
else:
    print(" Embeddings already exist in Milvus.")

# Load collection to memory for fast search
collection.load()

# Perform a simple search
def search_ad(query_embedding, top_k=5):
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["ad_id"]
    )
    return results

# Example query (using the first embedding as a query)
query_embedding = [embeddings[0]]
print(" Searching for similar ads...")
search_results = search_ad(query_embedding[0])

# Display results
for result in search_results[0]:
    print(f"Ad ID: {result.id}, Distance: {result.distance}")
