import pandas as pd
import numpy as np
import seaborn as sns
import re
from sklearn.preprocessing import OneHotEncoder

# ```room_info``` columns select
room_info = ['id', 'host_id', 'price',
             'property_type', 'room_type', 'accommodates', 'bathrooms', 'bathrooms_text', 'beds', 'amenities']

room_df = pd.read_csv('2025_Airbnb_NYC.csv', usecols = room_info, encoding = 'utf-8' )


# --- Shared preprocessing code ---
# Convert "id" from float to int, "price" from object to float
room_df['id'] = room_df['id'].astype(int)

room_df['price'] = room_df['price'].str.replace(r'[\$,]', '', regex=True)
room_df['price'] = room_df['price'].astype(float)


# --- Personal preprocessing code ---
# Convert "beds" from float to int
# Replace missing or non-bed values with median (assumed 1)
room_df['beds'] = room_df['beds'].fillna(0).astype(int)
room_df['beds'] = room_df['beds'].replace(0, 1)

# Clean up "bathrooms", "bathrooms_text" column:
# - Replace invalid or missing values with median (assumed 1)
room_df['bathrooms'] = room_df['bathrooms'].fillna(0)

def parse_baths(text):
    if pd.isna(text):
        return np.nan
    s = str(text).lower()
    m = re.search(r'(\d+(\.\d+)?)', s)
    if m:
        return float(m.group(1))
    if 'half' in s:
        return 0.5
    return np.nan

room_df['bathrooms_parsed'] = room_df['bathrooms_text'].apply(parse_baths)
mask_mismatch = room_df['bathrooms_parsed'].notna() & (room_df['bathrooms'] != room_df['bathrooms_parsed'])
room_df.loc[mask_mismatch, 'bathrooms'] = room_df.loc[mask_mismatch, 'bathrooms_parsed']
room_df = room_df.drop(columns=['bathrooms_parsed'])

room_df['bathrooms_text'] = room_df['bathrooms_text'].fillna(0)

room_df['is_shared'] = room_df['bathrooms_text'] \
    .str.contains('shared', case=False, na=False)

room_df['is_private'] = ~room_df['is_shared']

w_private = 1.0   # 전용 욕실 가중치
w_shared  = 0.5   # 공용 욕실 가중치

room_df['bath_score_mul'] = (
    room_df['bathrooms'] * np.where(room_df['is_private'], w_private, w_shared)
)

room_df['bathrooms'] = room_df['bathrooms'].replace(0.00, 1)
room_df['bath_score_mul'] = room_df['bath_score_mul'].replace(0.00, 1)

# Clean up "room_type", "property_type" column:
# 
def extract_structure(pt):
    pt_l = pt.strip().lower()
    if ' in ' in pt_l:
        return pt_l.split(' in ',1)[1].strip()
    if pt_l.startswith('entire '):
        return pt_l.replace('entire ','').strip()
    if pt_l.startswith('private room'):
        return pt_l.replace('private room','').strip()
    if pt_l.startswith('shared room'):
        return pt_l.replace('shared room','').strip()
    return pt_l

rt_cats = set(room_df['room_type'].str.strip().str.lower())
room_df['structure_type'] = room_df['property_type'].apply(lambda x: (
    x.strip().lower() if x.strip().lower() not in rt_cats
    else pd.NA
))

mask = room_df['structure_type'].notna()
room_df.loc[mask, 'structure_type'] = room_df.loc[mask, 'structure_type'].apply(extract_structure)

residential = {
    'rental unit','home','condo','townhouse','cottage',
    'bungalow','villa','vacation home','earthen home',
    'ranch','casa particular','tiny home','entire home/apt'
}
apartment_suite = {
    'guest suite','loft','serviced apartment','aparthotel',
    'private room'
}
hotel_lodging = {
    'hotel','boutique hotel','bed and breakfast',
    'resort','hostel','guesthouse','hotel room'
}

def map_category(row):
    pt = row['property_type'].strip().lower()
    rt = row['room_type'].strip().lower()
    st = row['structure_type']
    if rt in residential or pt in residential or (isinstance(st, str) and st in residential):
        return 'Residential'
    elif rt in apartment_suite or pt in apartment_suite or (isinstance(st, str) and st in apartment_suite):
        return 'Apartment_Suite'
    elif rt in hotel_lodging or pt in hotel_lodging or (isinstance(st, str) and st in hotel_lodging):
        return 'Hotel_Lodging'
    else:
        return 'Others'

room_df['structure_category'] = room_df.apply(map_category, axis=1)


'''
room_info = ['id', 'host_id', 'property_type', 'room_type', 'accommodates',
        'bathrooms', 'bathrooms_text', 'beds', 'price',
        'is_shared', 'is_private', 'bath_score_mul', 'structure_type',
        'structure_category']

room_info_filter = ['id', 'host_id', 'room_type', 'structure_type', 'structure_category'
        ,'accommodates', 'bath_score_mul', 'beds', 'price']

'''