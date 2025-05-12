import pandas as pd
from faker import Faker
import time

fake = Faker()

def generate_ads(num=100):
    data = [{'ad_id': i, 'text': fake.sentence(), 'category': fake.word()} for i in range(num)]
    df = pd.DataFrame(data)
    df.to_csv('data/ads.csv', index=False)
    print("Generated ads data successfully!")

def generate_users(num=50):
    data = [{'user_id': i, 'age': fake.random_int(min=18, max=65), 
             'gender': fake.random_element(elements=('Male', 'Female')),
             'interest': fake.word()} for i in range(num)]
    df = pd.DataFrame(data)
    df.to_csv('data/users.csv', index=False)
    print("Generated users data successfully!")

def simulate_realtime_generation():
    while True:
        generate_ads(10)
        generate_users(5)
        time.sleep(5)  # Wait 5 seconds to simulate real-time data ingestion

if __name__ == "__main__":
    simulate_realtime_generation()
 
