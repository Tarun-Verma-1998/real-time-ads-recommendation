import streamlit as st
import requests

st.title("🧠 Real-Time Ad Recommendation")
st.write("Enter a product-related query to get the top recommended ads:")

# Input
query = st.text_input("Type your ad query:", "50% off on mobile accessories")

if st.button("Get Recommendations"):
    try:
        # Send request to FastAPI
        response = requests.get("http://127.0.0.1:8000/recommend/", params={"query": query})
        data = response.json()

        st.success(f"Top Recommendations for: '{data['query']}'")

        for rec in data["recommendations"]:
            st.write(f"**{rec['rank']}. Ad ID:** {rec['ad_id']} — Score: {rec['score']:.4f}")

    except Exception as e:
        st.error("Error: Unable to connect to API. Make sure FastAPI is running.")
