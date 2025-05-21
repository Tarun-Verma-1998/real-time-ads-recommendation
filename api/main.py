from fastapi import FastAPI
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import xgboost as xgb
import numpy as np
import random

app = FastAPI()

# Load Milvus and model at startup
connections.connect("default", host="localhost", port="19530")
collection = Collection("ads_collection")
collection.load()

# Load models
ranker = xgb.XGBRanker()
ranker.load_model("models/xgb_ranking_model.json")
encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


@app.get("/recommend/")
def recommend(query: str = "Special discount on electronics"):
    # Encode the query
    query_embedding = encoder.encode([query])

    # Search Milvus
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param=search_params,
        limit=5,
        output_fields=["ad_id"]
    )

    # Build features and rank
    ad_ids = []
    features = []
    for hit in results[0]:
        distance = hit.distance
        dummy_feature = random.random()
        features.append([distance, dummy_feature])
        ad_ids.append(hit.id)

    features = np.array(features)
    scores = ranker.predict(features)

    ranked_ads = sorted(zip(ad_ids, scores), key=lambda x: x[1], reverse=True)
    ranked_result = [{"rank": i+1, "ad_id": ad_id, "score": float(score)} for i, (ad_id, score) in enumerate(ranked_ads)]

    return {"query": query, "recommendations": ranked_result}
