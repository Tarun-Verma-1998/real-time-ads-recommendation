import pandas as pd
import numpy as np
import xgboost as xgb
import os

# Step 1: Create fake user-ad interaction data
np.random.seed(42)
user_ids = np.arange(1, 6)   # 5 users
ad_ids = np.arange(1, 21)    # 20 ads

rows = []
for user in user_ids:
    for ad in np.random.choice(ad_ids, 10, replace=False):
        clicked = np.random.binomial(1, 0.3)  # 30% chance of click
        feature1 = np.random.rand()
        feature2 = np.random.rand()
        rows.append([user, ad, feature1, feature2, clicked])

df = pd.DataFrame(rows, columns=["user_id", "ad_id", "feature1", "feature2", "clicked"])
df.to_csv("data/train_data.csv", index=False)
print("✅ Training data saved to data/train_data.csv")

# Step 2: Prepare training data for XGBoost
X = df[["feature1", "feature2"]]
y = df["clicked"]

# Group by user for ranking
group_sizes = df.groupby("user_id").size().to_list()

# Step 3: Train XGBoost Ranker
model = xgb.XGBRanker(objective="rank:pairwise", learning_rate=0.1, max_depth=3, n_estimators=50)
model.fit(X, y, group=group_sizes)

# Step 4: Save model
model.save_model("models/xgb_ranking_model.json")
print(" XGBoost ranking model trained and saved.")
