from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import xgboost as xgb
import numpy as np
import random

# Step 1: Connect to Milvus and load collection
connections.connect("default", host="localhost", port="19530")
collection = Collection("ads_collection")
collection.load()

# Step 2: Load XGBoost model
ranker = xgb.XGBRanker()
ranker.load_model("models/xgb_ranking_model.json")

# Step 3: Load sentence transformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Step 4: Query text
query = "Special discount on new arrivals in electronics"
query_embedding = model.encode([query])  # Must be 2D list

# Step 5: Retrieve top-5 similar ads from Milvus
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
results = collection.search(
    data=query_embedding,
    anns_field="embedding",
    param=search_params,
    limit=5,
    output_fields=["ad_id"]
)

# Step 6: Build features for each result (distance, dummy feature)
ad_ids = []
features = []
for hit in results[0]:
    distance = hit.distance
    dummy_feature = random.random()  # Simulate additional user-ad relevance
    features.append([distance, dummy_feature])
    ad_ids.append(hit.id)

features = np.array(features)

# Step 7: Predict ranking scores
scores = ranker.predict(features)

# Step 8: Rank ads by score
ranked_ads = sorted(zip(ad_ids, scores), key=lambda x: x[1], reverse=True)

# Step 9: Display results
print("\n📊 Final Ranked Ads (after XGBoost reranking):")
for rank, (ad_id, score) in enumerate(ranked_ads, start=1):
    print(f"{rank}. Ad ID: {ad_id}, Score: {score:.4f}")
