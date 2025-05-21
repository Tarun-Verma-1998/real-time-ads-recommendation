from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import pickle
import numpy as np

# Step 1: Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Step 2: Load embeddings
with open('embeddings/ads_embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

# Step 3: Define schema
ad_id_field = FieldSchema(name="ad_id", dtype=DataType.INT64, is_primary=True, auto_id=True)
embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=len(embeddings[0]))
schema = CollectionSchema(fields=[ad_id_field, embedding_field], description="Ad Embeddings Collection")

# Step 4: Create collection (drop if exists)
collection_name = "ads_collection"
if utility.has_collection(collection_name):
    Collection(collection_name).drop()

collection = Collection(name=collection_name, schema=schema)

# Step 5: Insert embeddings
embedding_list = [list(vec) for vec in embeddings]
collection.insert([embedding_list])

# Step 6: Create index
collection.create_index(field_name="embedding", index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})
collection.load()

print("✅ Embeddings stored and indexed successfully in Milvus!")
