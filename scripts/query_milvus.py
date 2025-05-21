from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import numpy as np

# Step 1: Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Step 2: Load collection
collection = Collection("ads_collection")
collection.load()

# Step 3: Load embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Step 4: Input ad text
query_text = "Buy one get one free on fashion accessories!"  # Example new ad
query_embedding = model.encode([query_text])  # Must be 2D list

# Step 5: Search top-5 similar ads
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
results = collection.search(data=query_embedding, anns_field="embedding", param=search_params, limit=5, output_fields=["ad_id"])

# Step 6: Print results
print("\n🔍 Top 5 similar ads:")
for hit in results[0]:
    print(f"Ad ID: {hit.id}, Distance: {hit.distance:.4f}")
