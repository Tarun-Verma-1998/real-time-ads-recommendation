 Real-Time Ads Recommendation using XGBoost + T5

Project Overview

This project demonstrates a real-time ads recommendation pipeline using XGBoost and T5 embeddings. Due to the complexity of real-world deployment, the project simulates the pipeline, showcasing end-to-end integration using synthetic data.

Key Features:

Real-time data simulation and ingestion

Embedding generation using T5

Candidate retrieval using Milvus

Ads ranking using XGBoost

API for serving recommendations (FastAPI)

Interactive visualization using Streamlit

Containerized deployment using Docker and Docker Compose

Architecture:

Data Ingestion: Simulated ad and user data stored locally

Embedding Generation: T5 model for ad embeddings

Candidate Retrieval: Milvus to find similar ads

Ranking: XGBoost for ad ranking based on relevance

Serving: FastAPI to serve recommendations

Visualization: Streamlit to display ranked ads

Getting Started:
git clone https://github.com/Tarun-Verma-1998/real-time-ads-recommendation.git

Clone the repository:
git clone https://github.com/yourusername/real-time-ads-recommendation.git

Install dependencies:
pip install -r requirements.txt

Run the simulation pipeline:
docker-compose up

Contributing:

Feel free to fork the project and submit PRs for improvements!

