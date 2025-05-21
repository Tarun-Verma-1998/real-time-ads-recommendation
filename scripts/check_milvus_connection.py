from pymilvus import connections

# Corrected connection to Milvus
try:
    connections.connect("default", host="127.0.0.1", port="19530")
    print("Successfully connected to Milvus!")
except Exception as e:
    print(f"Connection failed: {e}")
