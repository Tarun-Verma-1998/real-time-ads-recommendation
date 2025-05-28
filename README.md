# Real-Time Ads Recommendation System (T5 + Milvus + XGBoost)

###  Project Overview

This project demonstrates a **real-time ads recommendation pipeline** that simulates end-to-end personalization using **T5 embeddings for semantic retrieval** and **XGBoost for ranking**. The system mirrors production-style architecture using synthetic user-ad interaction data.

It integrates real-time components, ranking models, vector search, and scalable inference to illustrate how platforms like marketplaces or ad servers deliver personalized content in milliseconds.

---

### 🔍 Key Features

-  **Real-time data simulation** with user and ad streams  
-  **Embedding generation** using pre-trained T5  
-  **Semantic retrieval** using Milvus (vector database)  
-  **Ranking** with XGBoost (CTR-style scoring)  
-  **Model serving** with FastAPI and Triton Inference Server  
-  **Visualization** of ranked ads via Streamlit UI  
-  **Containerized deployment** using Docker + Docker Compose  

---

###  System Architecture

```
[Simulated Data] → [T5 Embeddings] → [Milvus Retrieval] → [XGBoost Reranker] → [FastAPI API] → [Streamlit UI]
```

1. **Data Ingestion**: Simulates user + ad interactions  
2. **Embedding Layer**: Uses T5 model to embed ads  
3. **Candidate Retrieval**: Milvus returns top-k ads via vector similarity  
4. **Re-ranking**: XGBoost scores and sorts ads based on relevance  
5. **Serving**: FastAPI endpoint delivers ranked ads  
6. **Visualization**: Streamlit app displays final results  

---

###  Getting Started

```bash
# Clone the repository
git clone https://github.com/Tarun-Verma-1998/real-time-ads-recommendation.git

# Install Python dependencies
pip install -r requirements.txt

# Start the pipeline
docker-compose up
```

---

###  Contributing

Contributions are welcome! Feel free to fork this repo and open a PR to improve or extend the pipeline.

---

###  Why This Matters

This project simulates what real-world ad tech and personalization systems look like — from embedding-based retrieval to re-ranking and real-time serving.  
It’s a practical sandbox for ML engineers exploring **retrieval-augmented ranking**, **streaming recommendations**, and **low-latency inference pipelines**.
