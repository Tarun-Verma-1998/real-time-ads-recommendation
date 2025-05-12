from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle
import os

# Load the T5 embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load ad data
ad_file = 'data/ads.csv'
if not os.path.exists(ad_file):
    print(f"Error: {ad_file} not found. Please run the data simulation script first.")
    exit()

df = pd.read_csv(ad_file)

# Generate embeddings for ad text
print("Generating embeddings for ads...")
ad_texts = df['text'].tolist()
embeddings = model.encode(ad_texts)

# Save embeddings to a file
embedding_file = 'embeddings/ads_embeddings.pkl'
with open(embedding_file, 'wb') as f:
    pickle.dump(embeddings, f)

print(f"Embeddings saved to {embedding_file}")
 
