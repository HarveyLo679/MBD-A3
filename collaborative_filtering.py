from surprise import Dataset, Reader, SVD, accuracy
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
class CFRecommender:
    def __init__(self, data_path):
        self.data_path = data_path
        random.seed(42)
        np.random.seed(42) 
        
        self.svd = SVD()
        
        self.train()
        
    def load_df(self):
        df = pd.read_csv(self.data_path)
        df = df.dropna()
        df['User_id'] = df['User_id'].astype('int')
        df['year'] = df['year'].astype('int')
        df['month'] = df['month'].astype('int')
        df['day'] = df['day'].astype('int')
        
        self.df = df
        
    def preprocess_df(self):
        # Implicit feedback
        self.df['purchase'] = 1
        
        # Split into train and test sets 80/20
        train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=42)
        self.train_df = train_df
        self.test_df = test_df
        
        # Prepare data for SVD
        reader = Reader(rating_scale=(0, 1))
        train_data_surprise = Dataset.load_from_df(self.train_df[['User_id', 'itemDescription', 'purchase']], reader)
        self.trainset = train_data_surprise.build_full_trainset()
        self.testset = list(self.test_df[['User_id', 'itemDescription', 'purchase']].itertuples(index=False, name=None))
        
    def train(self):
        self.load_df()
        self.preprocess_df()
        
        self.svd.fit(self.trainset)
        
    def handle_cold_start(self, top_n=5): # New user
        # Return most frequently bought items
        popular_items = self.train_df['itemDescription'].value_counts().index[:top_n]
        return [(item, 0.0) for item in popular_items]
        
    def evaluate(self):
        return accuracy.rmse(self.svd.test(self.testset))
    
    def recommend(self, user_id, patterns=None, top_n=5, is_mock=False):
        if user_id not in self.train_df['User_id'].unique():
            print(f"Cold start for User ID {user_id}")
            return self.handle_cold_start(top_n)

        user_items = set(self.train_df[self.train_df['User_id'] == user_id]['itemDescription'])
        all_items = set(self.train_df['itemDescription'].unique())
        items_to_recommend = all_items - user_items

        # Collaborative Filtering Predictions
        recommendations = {}
        for item in items_to_recommend:
            try:
                pred = self.svd.predict(user_id, item)
                recommendations[item] = pred.est
            except:
                continue

        sorted_cf_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)

        if not patterns and not is_mock:
            return sorted_cf_recs[:top_n]

        # MOCKED frequent-pattern fallback (replace with your actual pattern engine)
        pattern_based = None
        if is_mock:
            pattern_based = self.mock_pattern_based(user_items)
        else: 
            pattern_based = patterns 
        
        for item, score in pattern_based:
            if item not in user_items and item not in recommendations:
                recommendations[item] = score

        final_sorted = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return final_sorted[:top_n]

    def mock_pattern_based(self, user_items):
        # Simulated rules: if user buys 'sugar' → recommend 'milk'
        rules = {
            'sugar': [('milk', 0.7)],
            'flour': [('eggs', 0.6)],
            'ketchup': [('mayonnaise', 0.5)],
        }
        recs = []
        for item in user_items:
            if item in rules:
                recs.extend(rules[item])
        return recs

if __name__ == "__main__":
    recommender = CFRecommender("data/Groceries data train.csv")

    while True:
        user_input = input("\nEnter User ID (or 'exit'): ").strip()
        if user_input.lower() == 'exit':
            break
        if not user_input.isdigit():
            print("Invalid User ID.")
            continue

        user_id = int(user_input)

        pattern_input = input("Use frequent patterns? (yes/no): ").strip().lower()
        use_patterns = pattern_input == 'yes'

        recommendations = recommender.recommend(user_id, is_mock=True)
        print("======================================================")
        print(f"Top 5 recommendations for User ID {user_id} ({'with' if use_patterns else 'without'} patterns):")
        for item, score in recommendations:
            print(f"{item}: Score = {score:.4f}")
     