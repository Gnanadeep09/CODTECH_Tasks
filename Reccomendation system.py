import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

# Load dataset (MovieLens 100k as an example)
url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
columns = ["user_id", "item_id", "rating", "timestamp"]
df = pd.read_csv(url, sep="\t", names=columns)

# Define reader with rating scale
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

# Use SVD (Singular Value Decomposition) for collaborative filtering
model = SVD()
cross_validate(model, data, cv=5, verbose=True)

# Train on full dataset
trainset = data.build_full_trainset()
model.fit(trainset)

# Make predictions for a specific user
user_id = str(196)
item_id = str(302)
prediction = model.predict(user_id, item_id)
print(f"Predicted rating for user {user_id} and item {item_id}: {prediction.est:.2f}")