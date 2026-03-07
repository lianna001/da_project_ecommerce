"""
generate_daily.py
GitHub Actions runs this script every day automatically.
It generates daily item stats and saves them to data/daily/
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

# ── Config ──────────────────────────────────────────
NUM_SELLERS    = 500
NUM_ITEMS      = 5000
AD_ITEM_RATIO  = 0.6
OUTPUT_DIR     = 'data/daily'
STATIC_DIR     = 'data/static'
# ────────────────────────────────────────────────────

CATEGORIES = [
    'Electronics', 'Fashion', 'Beauty', 'Food',
    'Sports', 'Home & Living', 'Books', 'Baby & Toys'
]

PRICE_RANGE = {
    'Electronics':   (50000,  1500000),
    'Fashion':       (10000,  300000),
    'Beauty':        (5000,   150000),
    'Food':          (3000,   80000),
    'Sports':        (15000,  500000),
    'Home & Living': (8000,   400000),
    'Books':         (8000,   50000),
    'Baby & Toys':   (10000,  200000),
}

AD_TYPES       = ['Recommendation', 'Category Banner']
SELLER_GRADES  = ['Basic', 'Regular', 'Power', 'Premium']
SELLER_WEIGHTS = [0.40, 0.30, 0.20, 0.10]


def make_sellers():
    np.random.seed(42)
    n = NUM_SELLERS
    return pd.DataFrame({
        'seller_id':         [f'S{str(i).zfill(4)}' for i in range(1, n+1)],
        'seller_grade':      np.random.choice(SELLER_GRADES, size=n, p=SELLER_WEIGHTS),
        'main_category':     np.random.choice(CATEGORIES, size=n),
        'joined_months_ago': np.random.randint(1, 60, size=n),
        'country':           np.random.choice(['KR','CN','US','JP'], size=n, p=[0.6,0.2,0.1,0.1]),
        'avg_rating':        np.round(np.random.uniform(3.0, 5.0, size=n), 1),
        'total_items':       np.random.randint(1, 200, size=n),
    })


def make_items(sellers_df):
    np.random.seed(7)
    n = NUM_ITEMS
    weights = sellers_df['total_items'].values.astype(float)
    weights /= weights.sum()
    assigned = sellers_df.sample(n=n, replace=True, weights=weights).reset_index(drop=True)
    categories = np.random.choice(CATEGORIES, size=n)
    prices = [
        round(random.randint(PRICE_RANGE[c][0]//1000, PRICE_RANGE[c][1]//1000) * 1000)
        for c in categories
    ]
    reg_dates = [
        (datetime(2024,1,1) + timedelta(days=random.randint(0,364))).strftime('%Y-%m-%d')
        for _ in range(n)
    ]
    return pd.DataFrame({
        'item_id':         [f'ITEM{str(i).zfill(5)}' for i in range(1, n+1)],
        'seller_id':       assigned['seller_id'].values,
        'seller_grade':    assigned['seller_grade'].values,
        'category':        categories,
        'price':           prices,
        'registered_date': reg_dates,
        'stock':           np.random.randint(0, 500, size=n),
        'avg_rating':      np.round(np.random.uniform(2.5, 5.0, size=n), 1),
        'review_count':    np.random.randint(0, 2000, size=n),
        'ab_group':        np.random.choice(['A','B'], size=n),
    })


def make_ad_campaigns(items_df):
    random.seed(42)
    campaigns = []
    ad_items = items_df.sample(frac=AD_ITEM_RATIO, random_state=42)
    campaign_id = 1
    for _, item in ad_items.iterrows():
        num_campaigns = random.choice([1, 1, 2])
        used_periods = []
        for _ in range(num_campaigns):
            start_offset = random.randint(0, 330)
            start = datetime(2025,1,1) + timedelta(days=start_offset)
            duration = random.randint(7, 60)
            end = start + timedelta(days=duration)
            overlap = any(not (end < s or start > e) for s, e in used_periods)
            if overlap:
                continue
            used_periods.append((start, end))
            grade_multiplier = {'Basic':1,'Regular':2,'Power':4,'Premium':8}
            budget = random.randint(50000,200000) * grade_multiplier.get(item['seller_grade'],1)
            campaigns.append({
                'campaign_id':  f'CAM{str(campaign_id).zfill(6)}',
                'item_id':      item['item_id'],
                'seller_id':    item['seller_id'],
                'ad_type':      random.choice(AD_TYPES),
                'start_date':   start.strftime('%Y-%m-%d'),
                'end_date':     end.strftime('%Y-%m-%d'),
                'duration_days':duration,
                'daily_budget': budget,
            })
            campaign_id += 1
    return pd.DataFrame(campaigns)


def make_daily_stats(date_str, items_df, campaigns_df):
    target_date = datetime.strptime(date_str, '%Y-%m-%d')
    active_ads = campaigns_df[
        (pd.to_datetime(campaigns_df['start_date']) <= target_date) &
        (pd.to_datetime(campaigns_df['end_date'])   >= target_date)
    ][['item_id','ad_type','daily_budget']].drop_duplicates('item_id')

    df = items_df.copy()
    df = df.merge(active_ads, on='item_id', how='left')
    df['ad_active']    = df['ad_type'].notna().astype(int)
    df['ad_type']      = df['ad_type'].fillna('None')
    df['daily_budget'] = df['daily_budget'].fillna(0)
    n = len(df)

    grade_imp  = {'Basic':1.0,'Regular':1.3,'Power':1.8,'Premium':2.5}
    base_imp   = np.random.randint(100, 1000, size=n)
    grade_mult = df['seller_grade'].map(grade_imp).values
    ad_boost   = np.where(df['ad_active']==1, np.random.uniform(2.0,5.0,size=n), 1.0)
    type_boost = np.where(df['ad_type']=='Recommendation',
                          np.random.uniform(1.2,1.5,size=n),
                          np.where(df['ad_type']=='Category Banner',
                                   np.random.uniform(1.5,3.0,size=n), 1.0))
    impressions = (base_imp * grade_mult * ad_boost * type_boost).astype(int)

    base_ctr = np.random.uniform(0.01, 0.05, size=n)
    ad_ctr   = np.where(df['ad_active']==1, base_ctr * np.random.uniform(1.3,2.0,size=n), base_ctr)
    ad_ctr   = np.clip(ad_ctr, 0, 0.30)
    clicks   = (impressions * ad_ctr).astype(int)

    rating_factor = (df['avg_rating'].values - 2.5) / 2.5
    review_factor = np.log1p(df['review_count'].values) / np.log1p(2000)
    cvr = np.clip(np.random.uniform(0.02,0.08,size=n) * (1 + 0.3*rating_factor + 0.2*review_factor), 0, 0.30)

    orders  = (clicks * cvr).astype(int)
    revenue = orders * df['price'].values

    stats = pd.DataFrame({
        'date':              date_str,
        'item_id':           df['item_id'].values,
        'seller_id':         df['seller_id'].values,
        'seller_grade':      df['seller_grade'].values,
        'category':          df['category'].values,
        'price':             df['price'].values,
        'ab_group':          df['ab_group'].values,
        'ad_active':         df['ad_active'].values,
        'ad_type':           df['ad_type'].values,
        'daily_budget':      df['daily_budget'].values,
        'impressions':       impressions,
        'clicks':            clicks,
        'orders':            orders,
        'revenue':           revenue,
        'ctr':               np.round(np.where(impressions>0, clicks/impressions, 0), 4),
        'conversion_rate':   np.round(np.where(clicks>0, orders/clicks, 0), 4),
        'revenue_per_click': np.round(np.where(clicks>0, revenue/clicks, 0), 0),
    })
    return stats[stats['impressions'] > 0].reset_index(drop=True)


def run():
    today = datetime.today().strftime('%Y-%m-%d')
    print(f'Generating data for {today}...')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)

    # Load or create static tables
    sellers_path   = f'{STATIC_DIR}/sellers.csv'
    items_path     = f'{STATIC_DIR}/items.csv'
    campaigns_path = f'{STATIC_DIR}/ad_campaigns.csv'

    if not os.path.exists(sellers_path):
        print('Creating static tables...')
        sellers_df   = make_sellers()
        items_df     = make_items(sellers_df)
        campaigns_df = make_ad_campaigns(items_df)
        sellers_df.to_csv(sellers_path, index=False)
        items_df.to_csv(items_path, index=False)
        campaigns_df.to_csv(campaigns_path, index=False)
        print(f'  → {len(sellers_df):,} sellers, {len(items_df):,} items, {len(campaigns_df):,} campaigns saved')
    else:
        items_df     = pd.read_csv(items_path)
        campaigns_df = pd.read_csv(campaigns_path)
        print('Static tables loaded.')

    # Generate today's stats
    stats = make_daily_stats(today, items_df, campaigns_df)
    output_path = f'{OUTPUT_DIR}/daily_stats_{today}.csv'
    stats.to_csv(output_path, index=False)

    print(f'Done! {len(stats):,} rows saved to {output_path}')
    print(f'  Ad active : {stats["ad_active"].sum():,} items')
    print(f'  Revenue   : {stats["revenue"].sum():,}')


if __name__ == '__main__':
    run()
