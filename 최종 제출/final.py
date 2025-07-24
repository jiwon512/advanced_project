# GIS 주요 데이터 2025_Airbnb_NYC_listings.csv, mappluto.geojson
# 1. 전처리 

import pandas as pd
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import re


df = pd.read_csv('/Users/com/Documents/GitHub/advanced_project/jiwon_project/csv_files/NY_Airbnb_original_df.csv')

# 전처리 1 - id int로 변경
df['id'] = df['id'].astype(int)

# booking_info
df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
df['instant_bookable'] =  df['instant_bookable'].map({'f':0, 't':1})
df['is_long_term'] = (df['minimum_nights'] >= 28).astype(int)

# amenities_info
def parse_amenities(x):
    try:
        return [a.strip().strip('"').strip("'") for a in ast.literal_eval(x)]
    except:
        return []
    
df['amenities'] = df['amenities'].apply(parse_amenities)
df['amenities_cnt'] = df['amenities'].apply(len)

# host_info
# neighborhood_overview 결측치 많아서 유무대체 
df['neighborhood_overview_exists'] = df['neighborhood_overview'].notnull().astype(int)

# name 글자수기준 중앙값으로 그룹
df['name_length'] = df['name'].fillna('').astype(str).apply(len)
med_length = df['name_length'].median()

def name_length_group(length, med):
    if length == 0:
        return 'empty'
    elif length > med:
        return 'long'
    else:
        return 'short_or_med'
    
df['name_length_group'] = df['name_length'].apply(lambda x: name_length_group(x, med_length))

# description 글자수기준(결측치 405) 평균으로 그룹
df['description_length'] = df['description'].fillna('').astype(str).apply(len)
avg_length = df['description_length'].mean()

def name_length_group(length, avg):
    if length == 0:
        return 'empty'
    elif length > avg:
        return 'long'
    else:
        return 'short_or_avg'
    
df['description_length_group'] = df['description_length'].apply(lambda x: name_length_group(x, avg_length))

# host_about (결측치8917) 평균(243) 중앙값(81) 중앙값기준으로 그룹
df['host_about_length'] = df['host_about'].fillna('').astype(str).apply(len)
med_length = df['host_about_length'].median()

def name_length_group(length, med):
    if length == 0:
        return 'empty'
    elif length > med:
        return 'long'
    else:
        return 'short_or_med'
df['host_about_length_group'] = df['host_about_length'].apply(lambda x: name_length_group(x, med_length))


#host_identity_verified/host_has_profile_pic /host_is_superhost  
# True / Flase 1과 0으로 대체 (결측치 20/20/350 0으로 대체함)
df['host_identity_verified']=df['host_identity_verified'].fillna('f').map({'t': 1, 'f': 0}).astype(int)

df['host_has_profile_pic']=df['host_has_profile_pic'].fillna('f').map({'t': 1, 'f': 0}).astype(int)

df['host_is_superhost']=df['host_is_superhost'].fillna('f').map({'t': 1, 'f': 0}).astype(int)

# host_response_time 결측치는 중앙값으로 치환후 점수
response_time_score_map = { 
    'within an hour': 4,
    'within a few hours': 3,
    'within a day': 2,
    'a few days or more': 1
}
df['host_response_time_score'] = df['host_response_time'].map(response_time_score_map)

# 2. response_time_score 컬럼의 중앙값 계산
med_score_for_fillna = df['host_response_time_score'].median()

# 3. response_time_score 컬럼의 NaN을 계산된 중앙값으로 대체 
df['host_response_time_score'] = df['host_response_time_score'].fillna(med_score_for_fillna)

# host_response_time 칼럼에는 여전히 nan값 존재함
# response_time_score 칼럼만 중앙값대체 


# host_response_rate 컬럼 %제외하고 중앙값으로 대체
df['host_response_rate'] = df['host_response_rate'].astype(str).str.replace('%', '').astype(float)/100
med_rate2 = df['host_response_rate'].median()
df['host_response_rate']= df['host_response_rate'].fillna(med_rate2)

# 4그룹으로 나눠 점수
conditions = [
    (df['host_response_rate'] <= 0.25),
    (df['host_response_rate'] > 0.25) & (df['host_response_rate'] <= 0.5),
    (df['host_response_rate'] > 0.5) & (df['host_response_rate'] <= 0.75),
    (df['host_response_rate'] > 0.75)
]

choices = [1, 2, 3, 4]

df['host_response_rate_score'] = np.select(conditions, choices)


# host_acceptance_rate 칼럼도 %제외하고 중앙값으로 대체 
df['host_acceptance_rate'] = df['host_acceptance_rate'].astype(str).str.replace('%', '').astype(float)/100
med_rate = df['host_acceptance_rate'].median()
df['host_acceptance_rate']= df['host_acceptance_rate'].fillna(med_rate)

conditions = [
    (df['host_acceptance_rate'] <= 0.25),
    (df['host_acceptance_rate'] > 0.25) & (df['host_acceptance_rate'] <= 0.5),
    (df['host_acceptance_rate'] > 0.5) & (df['host_acceptance_rate'] <= 0.75),
    (df['host_acceptance_rate'] > 0.75)
]

choices = [1, 2, 3, 4]

df['host_acceptance_rate_score'] = np.select(conditions, choices)

# host_location 칼럼 
# host_loc 존재?
df['host_location_boolean'] = df['host_location'].notnull().astype(int)
# host_loc in NY?
df['host_location_ny'] = df['host_location'].str.contains('New York', na=False).astype(int)



# === rooms_info ===
# --- Personal preprocessing code ---
# Convert "beds" from float to int
# Replace missing or non-bed values with median (assumed 1)
df['beds'] = df['beds'].fillna(0).astype(int)
df['beds'] = df['beds'].replace(0, 1)

# Clean up "bathrooms", "bathrooms_text" column:
# - Replace invalid or missing values with median (assumed 1)
df['bathrooms'] = df['bathrooms'].fillna(0)

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

df['bathrooms_parsed'] = df['bathrooms_text'].apply(parse_baths)
mask_mismatch = df['bathrooms_parsed'].notna() & (df['bathrooms'] != df['bathrooms_parsed'])
df.loc[mask_mismatch, 'bathrooms'] = df.loc[mask_mismatch, 'bathrooms_parsed']
df = df.drop(columns=['bathrooms_parsed'])

df['bathrooms_text'] = df['bathrooms_text'].fillna(0)

df['is_shared'] = df['bathrooms_text'] \
    .str.contains('shared', case=False, na=False)

df['is_private'] = ~df['is_shared']

w_private = 1.0   # 전용 욕실 가중치
w_shared  = 0.5   # 공용 욕실 가중치

df['bath_score_mul'] = (
    df['bathrooms'] * np.where(df['is_private'], w_private, w_shared)
)

df['bathrooms'] = df['bathrooms'].replace(0.00, 1)
df['bath_score_mul'] = df['bath_score_mul'].replace(0.00, 1)

# Clean up "room_type", "property_type" column:
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

rt_cats = set(df['room_type'].str.strip().str.lower())
df['room_structure_type'] = df['property_type'].apply(lambda x: (
    x.strip().lower() if x.strip().lower() not in rt_cats
    else pd.NA
))

mask = df['room_structure_type'].notna()
df.loc[mask, 'room_structure_type'] = df.loc[mask, 'room_structure_type'].apply(extract_structure)
df['room_structure_type'] = df['room_structure_type'].fillna('rental unit')

for col in [
    'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
    'review_scores_checkin', 'review_scores_communication',
    'review_scores_location', 'review_scores_value'
]:
    df[col].fillna(df[col].mean(), inplace=True)
    df[col] = df[col].round(2)

# host_since 년도로 바꾸기, 결측치 비율 0.09%-> 0으로 
df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
df['host_since'] = df['host_since'].dt.year
df['host_since'] = df['host_since'].fillna(0).astype(int)
df['host_since'] = df['host_since'].astype(int)

# last_review 년도로 바꾸기, 결측치 비율 30% -> 0으로
df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
df['last_review'] = df['last_review'].dt.year
df['last_review'] = df['last_review'].fillna(0).astype(int)
df['last_review'] = df['last_review'].astype(int)

# 비활성화 조건
cond1 = (df['last_review'] <= 2022) & (df['last_review'] != 0) & (df['estimated_occupancy_l365d'] == 0)
cond2 = (df['host_since'] <= 2022) & (df['number_of_reviews'] == 0) & (df['estimated_occupancy_l365d'] == 0)

# 활성화=1 비활성화=0
df['is_activate'] = np.select([cond1, cond2], [0, 0], default=1)

# 2 . POI 내려받기 ─────────────────────────────────────────────────────────────────────────────
import time
import json
import pandas as pd
import requests
import geopandas as gpd
from shapely.geometry import Point

# ─────────────────────────────────────────────────────────────────────────────
# 0) 원본 df, poi_tags, 그리고 bbox 계산
  # latitude, longitude 칼럼이 있어야 함
poi_tags = {
    'transport': {
        'amenity': ['bus_station','taxi'],
        'railway': ['station']
    },
    'infrastructure': {
        'amenity': ['police','hospital','pharmacy','restaurant','supermarket']
    },
    'tourism': {
        'tourism': ['viewpoint','museum','attraction'],
        'leisure': ['park']
    }
}
pad = 0.01
minx, maxx = df.longitude.min()-pad, df.longitude.max()+pad
miny, maxy = df.latitude.min()-pad, df.latitude.max()+pad
# ─────────────────────────────────────────────────────────────────────────────
# 1) 한 번에 bbox 내 모든 POI 내려받기 (Overpass bbox 쿼리)
OVERPASS_URL = "http://overpass-api.de/api/interpreter"
# build filters for bbox query
filters = ""
for grp in poi_tags.values():
    for key, vals in grp.items():
        for v in vals:
            filters += f'node["{key}"="{v}"]({miny},{minx},{maxy},{maxx});\n'
# full query
query = f"""
[out:json][timeout:180];
(
{filters}
);
out body;
"""
resp = requests.post(OVERPASS_URL, data={'data': query}, timeout=(5,300))
resp.raise_for_status()
data = resp.json().get('elements', [])
# ─────────────────────────────────────────────────────────────────────────────
# 2) GeoDataFrame 생성
pois = pd.DataFrame([
    {
      'lon': el['lon'],
      'lat': el['lat'],
      **el.get('tags',{})
    }
    for el in data
    if el['type']=='node' and 'lon' in el
])
gdf_pois = gpd.GeoDataFrame(
    pois,
    geometry=gpd.points_from_xy(pois.lon, pois.lat),
    crs="EPSG:4326"
).to_crs(epsg=3857)
# 원본 좌표도 GeoDataFrame
gdf_pts = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.longitude, df.latitude),
    crs="EPSG:4326"
).to_crs(epsg=3857)
sindex = gdf_pois.sindex
# ─────────────────────────────────────────────────────────────────────────────
# 3) 그룹별 카운트 함수
def count_group(pt, grp_map, radius=1000):
    buf = pt.buffer(radius)
    candidates = gdf_pois.iloc[list(sindex.intersection(buf.bounds))]
    cnt = 0
    for key, vals in grp_map.items():
        cnt += candidates[candidates[key].isin(vals)].shape[0]
    return cnt
# 4) 각 포인트별 count, df에 붙이기
for grp, tags in poi_tags.items():
    df[f"{grp}_count"] = [
        count_group(pt, tags, radius=1000)
        for pt in gdf_pts.geometry
    ]
# 5) 결과 확인
print(df[['transport_count','infrastructure_count','tourism_count']].head())

# 3. pca ─────────────────────────────────────────────────────────────────────────────
from sklearn.decomposition import PCA
poi_cols = ['transport_count','infrastructure_count','tourism_count']
pca = PCA(n_components=1)
# PCA fit → PC1 점수 생성
df['poi_pca1'] = pca.fit_transform(df[poi_cols].fillna(0))
# 설명 분산 비율 확인 (얼마나 데이터의 변동성을 담았는지)
print("Explained variance ratio (PC1):", pca.explained_variance_ratio_[0])
#poi_pca1 <0 poi 희박 지역, poi_pca1 > 0 poi 밀집지역


# 4. # 1단계: Airbnb 데이터 불러오기 & GeoDataFrame 변환
q1, q3 = df['poi_pca1'].quantile([0.25, 0.75])
def level(x):
    if x <= q1:   return '저개발'
    elif x >= q3: return '고개발'
    else:         return '보통'
df['poi_level'] = df['poi_pca1'].apply(level)
print(df['poi_level'].value_counts())

import pandas as pd
from shapely.geometry import Point
import geopandas as gpd

# 1. Airbnb CSV 불러오기 (lat, lon 칼럼 포함)
df_air = pd.read_csv("/Users/com/Documents/GitHub/advanced_project/jiwon_project/csv_files/NY_Airbnb_original_df.csv")

# 2. GeoDataFrame으로 변환
gdf_air = gpd.GeoDataFrame(
    df_air,
    geometry=[Point(xy) for xy in zip(df_air.longitude, df_air.latitude)],
    crs="EPSG:4326"
)

# 5. # 2단계: Overpass API로 POI 수집
# 분석할 POI 태그 그룹 정의
poi_tags = {
    'transport':    {'amenity': ['bus_station','taxi'], 'railway': ['station']},
    'infrastructure': {'amenity': ['police','hospital','pharmacy','supermarket']},
    'tourism':      {'tourism': ['viewpoint','museum','attraction'], 'leisure': ['park']}
}

# BBOX: Airbnb 전체 위경도 범위에 약간 패딩
pad = 0.01
minx, maxx = df_air.longitude.min() - pad, df_air.longitude.max() + pad
miny, maxy = df_air.latitude.min()  - pad, df_air.latitude.max()  + pad

import requests

# Overpass 쿼리 스트링 구축
filters = ""
for grp in poi_tags.values():
    for key, vals in grp.items():
        for v in vals:
            filters += f'node["{key}"="{v}"]({miny},{minx},{maxy},{maxx});\n'

query = f"""
[out:json][timeout:180];
(
{filters}
);
out body;
"""

# 요청
resp = requests.post("http://overpass-api.de/api/interpreter", data={'data': query})
resp.raise_for_status()
elements = resp.json()['elements']

# pandas DataFrame으로 변환
pois = pd.DataFrame([
    {'lon': el['lon'], 'lat': el['lat'], **el.get('tags', {})}
    for el in elements if el['type']=='node'
])

# GeoDataFrame
gdf_pois = gpd.GeoDataFrame(
    pois,
    geometry=gpd.points_from_xy(pois.lon, pois.lat),
    crs="EPSG:4326"
).to_crs(epsg=3857)  # 얘만 미터 단위로 바꿔두면 거리 계산이 편해집니다

# 6. # 3단계: 각 Airbnb 포인트마다 POI 개수 세어서 원본에 붙이기
# Airbnb 포인트도 미터 단위로
gdf_air_m = gdf_air.to_crs(epsg=3857)

# 공간 인덱스 생성
sidx = gdf_pois.sindex

def count_group(point, tag_map, radius=1000):
    """
    point: shapely Point (미터 단위 CRS)
    tag_map: poi_tags['transport']와 같은 dict
    radius: 반경(m)
    """
    buf = point.buffer(radius)
    # buffer bounds 로 후보 POI 인덱스 뽑기
    candidates = gdf_pois.iloc[list(sidx.intersection(buf.bounds))]
    cnt = 0
    # 각 태그 그룹별 count 합산
    for key, vals in tag_map.items():
        cnt += candidates[candidates[key].isin(vals)].shape[0]
    return cnt

# 결과를 저장할 칼럼 초기화
df_air['transport_count'] = 0
df_air['infrastructure_count'] = 0
df_air['tourism_count'] = 0

# point geometry 리스트
pts = gdf_air_m.geometry.tolist()

# 각 그룹별 count
for grp_name, tag_map in poi_tags.items():
    df_air[f"{grp_name}_count"] = [
        count_group(pt, tag_map, radius=1000)
        for pt in pts
    ]

# 이제 df_air 에 transport_count, infrastructure_count, tourism_count 가 생겼습니다
print(df_air[['transport_count','infrastructure_count','tourism_count']].head())

# 7. 1단계 df_poi = poi 전용 복제본 만들기 ─────────────────────────────────────────────────────────────────────────────
import pandas as pd

# 1) Airbnb 데이터 불러오기
df_air = pd.read_csv("/Users/com/Documents/GitHub/advanced_project/jiwon_project/csv_files/NY_Airbnb_original_df.csv")

# 2) POI 전용 복제본 만들기
df_poi = df_air.copy()

import pandas as pd
import geopandas as gpd
import requests
import numpy as np
from shapely.geometry import Point


# Airbnb 원본 불러오기
df_air = pd.read_csv("/Users/com/Documents/GitHub/advanced_project/jiwon_project/csv_files/NY_Airbnb_original_df.csv")

# GeoDataFrame으로 변환
gdf_air = gpd.GeoDataFrame(
    df_air,
    geometry=[Point(xy) for xy in zip(df_air.longitude, df_air.latitude)],
    crs="EPSG:4326"
)

# 분석할 POI 태그 정의
poi_tags = {
    'transport':     {'amenity': ['bus_station','taxi'], 'railway': ['station']},
    'infrastructure':{'amenity': ['police','hospital','pharmacy','supermarket']},
    'tourism':       {'tourism': ['viewpoint','museum','attraction'], 'leisure': ['park']}
}

# bbox 계산 (Padding ±0.01도)
pad = 0.01
minx, maxx = df_air.longitude.min()-pad, df_air.longitude.max()+pad
miny, maxy = df_air.latitude.min()-pad,  df_air.latitude.max()+pad

# Overpass 쿼리 생성
filters = ""
for grp in poi_tags.values():
    for key, vals in grp.items():
        for v in vals:
            filters += f'node["{key}"="{v}"]({miny},{minx},{maxy},{maxx});\n'

query = f"""
[out:json][timeout:180];
(
{filters}
);
out body;
"""
resp = requests.post("http://overpass-api.de/api/interpreter", data={'data': query})
resp.raise_for_status()
elements = resp.json().get("elements", [])


# pandas DataFrame으로
pois = pd.DataFrame([
    {'lon':el['lon'], 'lat':el['lat'], **el.get('tags',{})}
    for el in elements if el['type']=="node" and 'lon' in el
])

# GeoDataFrame (EPSG:3857 로 변환해서 거리 계산 준비)
gdf_pois = gpd.GeoDataFrame(
    pois,
    geometry=gpd.points_from_xy(pois.lon, pois.lat),
    crs="EPSG:4326"
).to_crs(epsg=3857)


# Airbnb 포인트도 EPSG:3857 으로 변환
gdf_air_m = gdf_air.to_crs(epsg=3857)
sidx = gdf_pois.sindex

def count_group(pt, tag_map, radius=1000):
    buf = pt.buffer(radius)
    candidates = gdf_pois.iloc[list(sidx.intersection(buf.bounds))]
    cnt = 0
    for key, vals in tag_map.items():
        cnt += candidates[candidates[key].isin(vals)].shape[0]
    return cnt

# df_air 에 POI 카운트 칼럼 초기화 & 계산
for grp, tags in poi_tags.items():
    df_air[f"{grp}_count"] = [
        count_group(pt, tags) 
        for pt in gdf_air_m.geometry
    ]

# 8. # 1단계. PCA 수행 & 라벨 생성─────────────────────────────────────────────────────────────────────────────
# 1-1) POI 카운트가 붙어있는 Airbnb DataFrame
df_air.columns  # transport_count, infrastructure_count, tourism_count 가 보이면 OK!

# 1-2) PCA 전용 복제본
df_poi = df_air.copy()

from sklearn.decomposition import PCA

# 사용할 POI 칼럼
poi_cols = ['transport_count','infrastructure_count','tourism_count']

# PCA 모델 생성
pca = PCA(n_components=1)

# 2-1) 첫 번째 주성분 계산해서 poi_pca1에 저장
df_poi['poi_pca1'] = pca.fit_transform(df_poi[poi_cols].fillna(0).values)

# 2-2) 발달/미발달 라벨 생성
df_poi['poi_cat'] = df_poi['poi_pca1'].apply(lambda x: '발달' if x>0 else '미발달')

# 2-3) 설명 분산 비율 확인 (보통 0.7 이상이면 PC1 하나로도 충분)
print("Explained variance ratio (PC1):", pca.explained_variance_ratio_[0])

# 2-4) 결과 샘플 확인
df_poi[['transport_count','infrastructure_count','tourism_count','poi_pca1','poi_cat']].head()


# 9. 2단계. # df_price를 만들어보자! ─────────────────────────────────────────────────────────────────────────────
# df_air 는 이미 'latitude','longitude','price' 칼럼을 가지고 있습니다
df_price = df_air[['id','latitude','longitude','price']].copy()

# 가격이 없는 행 제거 (필요시)
df_price = df_price.dropna(subset=['price'])

# 확인
print(df_price.head())

import pandas as pd

# 1) 문자열 형태의 price → 숫자(float)로 변환
#    • 달러 기호($)와 콤마(,) 제거
#    • 숫자로 변환, 변환 불가 값은 NaN
df_price['price'] = (
    df_price['price']
      .replace(r'[\$,]', '', regex=True)   # $ 와 , 제거
)
df_price['price'] = pd.to_numeric(df_price['price'], errors='coerce')

# 2) 변환 확인
print(df_price['price'].dtype)  # float64 여야 OK
print(df_price[['price']].head())

import branca.colormap as cm

vmin, vmax = df_price.price.min(), df_price.price.max()
price_cmap = cm.LinearColormap(
    ['blue','green','yellow','orange','red'],
    vmin=vmin, vmax=vmax,
    caption="Airbnb Price (USD)"
)

# 10. 최종 # 마지막 최종 본 poi, air bnb 가격 분포도, 지역별 가격 분포도 ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import geopandas as gpd
import folium
import requests
from folium.plugins import MarkerCluster
import branca.colormap as cm
from shapely.geometry import Point

# ── 0) Airbnb 데이터 불러와 GeoDataFrame으로 변환 ─────────────────────
df_air = pd.read_csv("/Users/com/Documents/GitHub/advanced_project/jiwon_project/csv_files/NY_Airbnb_original_df.csv")
df_air["price"] = df_air["price"].replace(r"[\$,]", "", regex=True).astype(float)

gdf_air = gpd.GeoDataFrame(
    df_air,
    geometry=gpd.points_from_xy(df_air.longitude, df_air.latitude),
    crs="EPSG:4326"
)

# ── 1) CD(Community District) 경계 셰이프 불러와서 병합 ─────────────
gdf_pluto = gpd.read_file("./mappluto.geojson").to_crs(epsg=4326)
gdf_cd = (
    gdf_pluto[["CD","geometry"]]
      .dissolve(by="CD")
      .reset_index()
)

# ── 2) Airbnb 포인트를 CD 폴리곤에 매핑 → CD별 평균 가격 집계 ─────────
air_cd = gpd.sjoin(gdf_air, gdf_cd, how="inner", predicate="within")
cd_stats = (
    air_cd.groupby("CD")["price"]
          .mean()
          .reset_index(name="avg_price")
)

# ── 3) POI 정보 불러오기 (Overpass API) ──────────────────────────────
poi_tags = {
    'transport':     {'amenity': ['bus_station','taxi'], 'railway': ['station']},
    'infrastructure':{'amenity': ['police','hospital','pharmacy','supermarket']},
    'tourism':       {'tourism': ['viewpoint','museum','attraction'], 'leisure': ['park']}
}
pad = 0.01
minx, maxx = df_air.longitude.min()-pad, df_air.longitude.max()+pad
miny, maxy = df_air.latitude.min()-pad, df_air.latitude.max()+pad

# Overpass bbox 쿼리 생성
filters = ""
for grp in poi_tags.values():
    for key,vals in grp.items():
        for v in vals:
            filters += f'node["{key}"="{v}"]({miny},{minx},{maxy},{maxx});\n'

query = f"""
[out:json][timeout:180];
(
{filters}
);
out body;
"""
resp = requests.post("http://overpass-api.de/api/interpreter", data={'data': query})
elements = resp.json().get("elements", [])

# DataFrame → GeoDataFrame
pois = pd.DataFrame([{
    'lat':e['lat'], 'lon':e['lon'], **e.get('tags',{})
} for e in elements if e['type']=='node'])
gdf_pois = gpd.GeoDataFrame(
    pois,
    geometry=gpd.points_from_xy(pois.lon, pois.lat),
    crs="EPSG:4326"
)

# ── 4) 가격 컬러맵 정의 (5–95% 트리밍) ──────────────────────────────
vmin, vmax = df_air.price.quantile([0.05, 0.95]).values
mid = (vmin+vmax)/2
price_cmap = cm.LinearColormap(
    ["green","yellow","red"],
    index=[vmin, mid, vmax],
    vmin=vmin, vmax=vmax,
    caption="Trimmed Airbnb Price (5–95%)"
)

# ── 5) Folium 지도 생성 ─────────────────────────────────────────────
m = folium.Map(
    location=[gdf_air.latitude.mean(), gdf_air.longitude.mean()],
    zoom_start=12,
    tiles="CartoDB positron"
)
m.add_child(price_cmap)

# ── 6) Choropleth: CD별 평균 가격 ──────────────────────────────────
folium.Choropleth(
    geo_data=gdf_cd.__geo_interface__,
    data=cd_stats,
    columns=["CD","avg_price"],
    key_on="feature.properties.CD",
    fill_color="YlOrRd",
    fill_opacity=0.6,
    line_opacity=0.2,
    legend_name="Avg Airbnb Price by CD"
).add_to(m)

# ── 7) POI 레이어: 그룹별 색·크기 구분 ──────────────────────────────
colors = {'transport':'blue','infrastructure':'green','tourism':'purple'}
for grp in poi_tags:
    layer = folium.FeatureGroup(name=f"POI ({grp})", show=False)
    for _, pt in gdf_pois.iterrows():
        # 각 노드의 태그 중 하나라도 grp 범위에 있으면
        if any(k in pt.index and pt[k] in poi_tags[grp].get(k,[]) for k in poi_tags[grp]):
            folium.CircleMarker(
                location=[pt.lat, pt.lon],
                radius=4,
                color=colors[grp],
                fill=True, fill_opacity=0.7
            ).add_to(layer)
    m.add_child(layer)

# ── 8) Airbnb Listings MarkerCluster ───────────────────────────────
mc = MarkerCluster(name="Airbnb Listings").add_to(m)
for _, r in gdf_air.iterrows():
    folium.CircleMarker(
        location=[r.latitude, r.longitude],
        radius=3,
        color=price_cmap(r.price),
        fill=True,
        fill_color=price_cmap(r.price),
        fill_opacity=0.7,
        popup=f"${r.price:.0f}"
    ).add_to(mc)

# ── 9) 레이어 토글 ─────────────────────────────────────────────────
folium.LayerControl(collapsed=False).add_to(m)

# ── 10) 지도 렌더링 ────────────────────────────────────────────────
m





