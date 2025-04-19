import pandas as pd
import numpy as np
from datetime import timedelta
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings("ignore")

class PatternRecommender:
    def __init__(self, raw_df):
        self.raw_df = self.preprocess_raw_data(raw_df)

    def preprocess_raw_data(self, raw_df):
        df = raw_df.copy()
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        df = df.dropna(subset=['user_id', 'itemdescription', 'date', 'year', 'month', 'day'])
        df['user_id'] = df['user_id'].astype(int)
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
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
        if not user_items:
            return [], pd.DataFrame()

        rules = rules.copy()
        for col in ['rec_value', 'surprise_score']:
            min_val = rules[col].min()
            max_val = rules[col].max()
            rules[col + '_scaled'] = (
                (rules[col] - min_val) / (max_val - min_val)
                if max_val > min_val else 0
            )

        def match_strength(rule):
            antecedents = set(rule['antecedents'])
            return len(antecedents.intersection(user_items)) / len(antecedents) if antecedents else 0

        rules['match_strength'] = rules.apply(match_strength, axis=1)
        rules['user_rule_score'] = rules['match_strength'] * (
            0.6 * rules['rec_value_scaled'] + 0.4 * rules['surprise_score_scaled']
        )

        matched_rules = rules[rules['match_strength'] > 0].copy()
        if matched_rules.empty:
            return [], pd.DataFrame()

        matched_rules = matched_rules.sort_values(by='user_rule_score', ascending=False)
        top_rules = matched_rules.head(top_k)

        recommended_items = list(pd.unique([
            item for items in top_rules['consequents'] for item in items if item not in user_items
        ]))

        if len(recommended_items) < top_k:
            popular_items = self.raw_df['itemdescription'].value_counts().index.tolist()
            for item in popular_items:
                if item not in recommended_items and item not in user_items:
                    recommended_items.append(item)
                if len(recommended_items) == top_k:
                    break

        return recommended_items[:top_k], top_rules

    def raw_data_mining(self, user_id=None, recent_days=None, min_support=0.0015, top_k=5, pre_mined_rules=None):
        user_id_str = str(user_id) if user_id is not None else None
        user_in_data = user_id_str in self.raw_df['user_id'].astype(str).unique() if user_id_str is not None else False

        def get_default_recommendations(rules, top_items, default_score=5.0):
            fallback = []
            exploded = rules.explode('consequents') if not rules.empty else pd.DataFrame()
            for item in top_items:
                match = exploded[exploded['consequents'] == item]
                score = round(match['rec_value'].max(), 5) if not match.empty else default_score
                fallback.append((item, score))
            return fallback

        if user_in_data:
            filtered_df, msg = self.filter_data(recent_days=recent_days, user_id=user_id)
            if filtered_df is None or filtered_df.empty:
                rules_subset = pre_mined_rules.sort_values(by='rec_value', ascending=False).head(top_k) if pre_mined_rules is not None else pd.DataFrame()
                top_items = self.raw_df['itemdescription'].value_counts().head(top_k).index.tolist()
                fallback_items = get_default_recommendations(rules_subset, top_items)
                return {
                    'mode': 'user mode (but no records)',
                    'recommended_items': fallback_items,
                    'reason': msg,
                    'top_rules': rules_subset.copy() if not rules_subset.empty else pd.DataFrame()
                }
        else:
            filtered_df, msg = self.filter_data(recent_days=recent_days)
            if filtered_df is None or filtered_df.empty:
                rules_subset = pre_mined_rules.sort_values(by='rec_value', ascending=False).head(top_k) if pre_mined_rules is not None else pd.DataFrame()
                top_items = self.raw_df['itemdescription'].value_counts().head(top_k).index.tolist()
                fallback_items = get_default_recommendations(rules_subset, top_items)
                return {
                    'mode': 'global mode (no data in recent_days)',
                    'recommended_items': fallback_items,
                    'reason': msg,
                    'top_rules': rules_subset.copy() if not rules_subset.empty else pd.DataFrame()
                }

        rules = pre_mined_rules if pre_mined_rules is not None else self.mining_patterns(filtered_df, min_support=min_support)

        user_items = (
            filtered_df[filtered_df['user_id'].astype(str) == str(user_id)]['itemdescription'].unique().tolist()
            if user_id is not None and user_in_data else []
        )

        recommended_items, top_rules = self.recommend_from_rules(rules, user_items, top_k=top_k)

        item_scores = {}
        for _, row in top_rules.iterrows():
            for item in row['consequents']:
                if item not in user_items:
                    item_scores[item] = row['rec_value']

        result = {}

        if not recommended_items:
            top_items = filtered_df['itemdescription'].value_counts().head(top_k).index.tolist()
            fallback_items = get_default_recommendations(rules, top_items)
            result = {
                'mode': 'user mode' if user_in_data else 'global mode',
                'recommended_items': fallback_items,
                'reason': 'No matched rules, fallback to popular items',
                'top_rules': rules.sort_values(by='rec_value', ascending=False).head(top_k).copy()
            }
        else:
            result = {
                'mode': 'user mode with rules' if user_in_data else 'global mode with rules',
                'recommended_items': [
                    (item, round(item_scores[item], 5)) if item in item_scores else (item, 5.0)
                    for item in recommended_items],
                'reason': 'Rules matched and used',
                'top_rules': top_rules.copy()
            }

        return result



