import pandas as pd
import numpy as np
from datetime import timedelta
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings("ignore")
import pickle

class PatternRecommender:
    def __init__(self, data_path):
        raw_df = pd.read_csv(data_path)
        self.raw_df = self.preprocess_raw_data(raw_df)

    def preprocess_raw_data(self, raw_df):
        df = raw_df.copy()
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        df = df.dropna(subset=['user_id', 'itemdescription', 'date', 'year', 'month', 'day'])
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        df['User_id'] = df['User_id'].astype('int')
        return df

    def filter_data(self, recent_days=None, user_id=None):
        df = self.raw_df.copy()
        if recent_days is not None:
            latest_date = df['date'].max()
            cutoff_date = latest_date - timedelta(days=recent_days)
            df = df[df['date'] >= cutoff_date]

        if user_id is not None:
            user_id = str(user_id)
            if user_id not in df['user_id'].astype(str).unique():
                return None, f"User {user_id} not found in selected data"
            df = df[df['user_id'].astype(str) == user_id]
            return df, f"User mode with recent {recent_days} days" if recent_days is not None else "User mode with full data"

        return df, f"Global mode with recent {recent_days} days" if recent_days is not None else "Global mode with full data"

    def mining_patterns(self, df, min_support=0.0015):
        transactions = df.groupby(['user_id', 'date'])['itemdescription'].apply(list).tolist()
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        basket = pd.DataFrame(te_ary, columns=te.columns_)

        frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

        # Custom metrics
        rules['combo_score'] = rules['support'] * rules['confidence'] * rules['lift']
        rules['surprise_score'] = rules['lift'] * np.log2(rules['confidence'] + 1e-10)
        rules['rec_value'] = (rules['confidence'] * rules['lift']) / rules['consequents'].apply(lambda x: len(x))
        rules['conf_scaled'] = rules['confidence'] / rules['confidence'].max()
        rules['lift_scaled'] = rules['lift'] / rules['lift'].max()
        rules['lev_scaled'] = rules['leverage'] / rules['leverage'].max()
        rules['NUS'] = rules['antecedents'].apply(lambda x: len(x))

        return rules

    def recommend_from_rules(self, rules, user_items, top_k=5):
        user_items = set(user_items)
        matched_rules = rules[rules['antecedents'].apply(lambda x: x.issubset(user_items))].copy()
        if matched_rules.empty:
            return [], pd.DataFrame()

        matched_rules = matched_rules.sort_values(by=['combo_score', 'surprise_score', 'rec_value'], ascending=False) # combo_score, then surprise_score, then rec_value
        top_rules = matched_rules.head(top_k)
        recommended_items = list(pd.unique([item for items in top_rules['consequents'] for item in items if item not in user_items]))
        return recommended_items[:top_k], top_rules

    def recommend(self, user_id=None, recent_days=None, min_support=0.0015, top_k=5, is_pre_mined_rules=True):
        user_in_data = user_id in self.raw_df['user_id'].unique() if user_id is not None else False

        if user_in_data:
            filtered_df, msg = self.filter_data(recent_days=recent_days, user_id=user_id)
            if filtered_df is None or filtered_df.empty:
                return {
                    'mode': 'user mode (but no records)',
                    'recommended_items': self.raw_df['itemdescription'].value_counts().head(top_k).index.tolist(),
                    'reason': msg
                }
        else:
            filtered_df, msg = self.filter_data(recent_days=recent_days)
            if filtered_df is None or filtered_df.empty:
                return {
                    'mode': 'global mode (no data in recent_days)',
                    'recommended_items': self.raw_df['itemdescription'].value_counts().head(top_k).index.tolist(),
                    'reason': msg
                }

        if is_pre_mined_rules:
            rules = None
            with open("saved_rules.pkl", "rb") as f:
                rules = pickle.load(f)
        else:
            rules = self.mining_patterns(filtered_df, min_support=min_support)

        user_items = (
            filtered_df[filtered_df['user_id'] == user_id]['itemdescription'].unique().tolist()
            if user_id is not None and user_in_data else []
        )
        recommended_items, top_rules = self.recommend_from_rules(rules, user_items, top_k=top_k)

        if not recommended_items:
            fallback_items = filtered_df['itemdescription'].value_counts().head(top_k).index.tolist()
            return {
                'mode': 'user mode' if user_in_data else 'global mode',
                'recommended_items': fallback_items,
                'reason': 'No matched rules, fallback to popular items',
                'top_rules': rules.sort_values(by='rec_value', ascending=False).head(top_k)[[
                    'antecedents', 'consequents', 'support', 'confidence', 'lift',
                    'combo_score', 'surprise_score', 'rec_value', 'NUS']]
            }

        result = {
            'mode': 'user mode with rules' if user_in_data else 'global mode with rules',
            'recommended_items': recommended_items,
            'reason': 'Rules matched and used',
            'top_rules': top_rules[[
                'antecedents', 'consequents', 'support', 'confidence', 'lift',
                'combo_score', 'surprise_score', 'rec_value', 'NUS']]
        }

        if user_id is not None and user_in_data:
            result['user_match_score'] = result['top_rules']

        return result
