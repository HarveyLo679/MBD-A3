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
        self.df = pd.read_csv(self.data_path)
        
    def preprocess_df(self):
        # Implicit feedback
        self.df['purchase'] = 1
        
        # Split into train and test sets 80/20
        train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=42)
        self.train_df = train_df
        self.test_df = test_df
        
        # Prepare data for SVD
        reader = Reader(rating_scale=(0, 1))
        train_data_surprise = Dataset.load_from_df(self.train_df[['user_id', 'item_description', 'purchase']], reader)
        self.trainset = train_data_surprise.build_full_trainset()
        self.testset = list(self.test_df[['user_id', 'item_description', 'purchase']].itertuples(index=False, name=None))
        
    def train(self):
        self.load_df()
        self.preprocess_df()
        
        self.svd.fit(self.trainset)
        
    def handle_cold_start(self, top_n=5): # New user
        # Return most frequently bought items
        return self.train_df['item_description'].value_counts().index[:top_n]
        
    def evaluate(self):
        return accuracy.rmse(self.svd.test(self.testset))
    
    def recommend(self, user_id, patterns=None, top_n=5, is_mock=False):
        """
        Recommend items for a given user_id.
        :param user_id: ID of the user to recommend items for
        :param patterns: List of tuple patterns to use for recommendations
        :param top_n: Number of desired recommendations to return
        :param is_mock: If True, use mock pattern-based recommendations"""
        
        if user_id not in self.train_df['user_id'].unique():
            return self.handle_cold_start(top_n)

        user_items = set(self.train_df[self.train_df['user_id'] == user_id]['item_description'])
        all_items = set(self.train_df['item_description'].unique())
        items_to_recommend = all_items - user_items

        # Collaborative Filtering Predictions
        recommendations = {}
        for item in items_to_recommend:
            try:
                pred = self.svd.predict(user_id, item)
                recommendations[item] = pred.est
            except:
                continue

        top_k_from_cf = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]

        if not patterns and not is_mock:
            return top_k_from_cf

        from_pattern_mining = None
        if is_mock:
            from_pattern_mining = self.mock_pattern_based(user_items)
        else: 
            from_pattern_mining = patterns 
        
        
        # Filter out items already bought by the user and items already in CF recommendations
        from_pattern_mining_filtered = [
            (item, score) for item, score in from_pattern_mining if item not in user_items and item not in top_k_from_cf
        ]
        
        # take half of the recommendations from CF and half from pattern mining, neglecting the score
        final_results = []
        final_results.extend(top_k_from_cf[:top_n // 2])
        final_results.extend(from_pattern_mining_filtered[:top_n // 2])
        
        return final_results

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
    recommender = CFRecommender("data/train.csv")

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
     