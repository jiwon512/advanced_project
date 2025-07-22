import pandas as pd
import pingouin as pg
import lightgbm as lgb
import numpy as np
import shap
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.compose import ColumnTransformer
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

csv_path = 'outlier_removed.csv'    # 여기에 absolute path
# CSV 읽기
df = pd.read_csv(
    csv_path,
    header=0,        # 첫 줄을 컬럼명으로 사용
    index_col='id',  # 인덱스 컬럼으로 id 지정
    encoding='utf-8-sig'
)
# 데이터 확인
df


import time
import json
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


from sklearn.decomposition import PCA
poi_cols = ['transport_count','infrastructure_count','tourism_count']
pca = PCA(n_components=1)
# PCA fit → PC1 점수 생성
df['poi_pca1'] = pca.fit_transform(df[poi_cols].fillna(0))
# 설명 분산 비율 확인 (얼마나 데이터의 변동성을 담았는지)
print("Explained variance ratio (PC1):", pca.explained_variance_ratio_[0])
#poi_pca1 <0 poi 희박 지역, poi_pca1 > 0 poi 밀집지역

'''
room_new_type 기준으로 필수 amenity와 필요 amenity 갖추고 있는 지수(점수로 표현)

공통 amenity (필수):
['Carbon monoxide alarm', 'Essentials', 'Hangers', 'Smoke alarm', 'Wifi']

high 특화 amenity:
['Air conditioning', 'Building staff', 'Elevator', 'Gym', 'Heating', 'Paid parking off premises', 'Shampoo']

low-mid 특화 amenity:
['Cleaning products', 'Dining table', 'Exterior security cameras on property', 'Free street parking', 'Freezer', 'Laundromat nearby', 'Lock on bedroom door', 'Microwave']

mid 특화 amenity:
['Cooking basics', 'Kitchen', 'Oven']

upper-mid 특화 amenity:
['Bathtub', 'Cleaning products', 'Cooking basics', 'Dishes and silverware', 'Elevator', 'Freezer']
'''


import ast

# 기준 Amenity 딕셔너리 정의
common_amenities = ['Carbon monoxide alarm', 'Essentials', 'Hangers', 'Smoke alarm', 'Wifi']

type_amenity_dict = {
    'high': ['Air conditioning', 'Building staff', 'Elevator', 'Gym', 'Heating', 'Paid parking off premises', 'Shampoo'],
    'low-mid': ['Cleaning products', 'Dining table', 'Exterior security cameras on property', 'Free street parking', 
                'Freezer', 'Laundromat nearby', 'Lock on bedroom door', 'Microwave'],
    'mid': ['Cooking basics', 'Kitchen', 'Oven'],
    'upper-mid': ['Bathtub', 'Cleaning products', 'Cooking basics', 'Dishes and silverware', 'Elevator', 'Freezer']}

# amenities 문자열 → 리스트로 파싱
def parse_amenities(row):
    try:
        return ast.literal_eval(row)
    except:
        return []

df['parsed_amenities'] = df['amenities'].apply(parse_amenities)

# amenity 매칭 점수 계산 함수
def calc_match_score(row):
    amenities = row['parsed_amenities']
    room_type = row['room_new_type']  # ← 미리 이 컬럼 만들어져 있어야 함
    
    # 공통 어매니티 일치 비율
    common_match = sum(1 for a in amenities if a in common_amenities) / len(common_amenities)
    
    # room type 별 특화 어매니티 일치 비율
    type_amenities = type_amenity_dict.get(room_type, [])
    if type_amenities:
        type_match = sum(1 for a in amenities if a in type_amenities) / len(type_amenities)
    else:
        type_match = 0.0
    
    return pd.Series({
        'common_amenity_score': round(common_match, 3),
        'type_amenity_score': round(type_match, 3)})

# 점수 컬럼 추가
df[['common_amenity_score', 'type_amenity_score']] = df.apply(calc_match_score, axis=1)

# 점수 해석을 위한 요약 출력
print(df[['room_new_type', 'common_amenity_score', 'type_amenity_score']].groupby('room_new_type').mean().round(3))




# 위치데이터 카운트 변수들 정규성/등분산성

from scipy.stats import shapiro, levene

Location = ['transport_count', 'infrastructure_count', 'tourism_count', 'poi_pca1']
TARGET = 'host_is_superhost'

for col in Location:
    print(f"\n 변수: {col}")

    # 정규성 검정 (랜덤 샘플링)
    group1 = df[df[TARGET]==1][col].dropna()
    group0 = df[df[TARGET]==0][col].dropna()
    
    n1 = min(5000, len(group1))
    n0 = min(5000, len(group0))

    stat1, p1 = shapiro(group1.sample(n1, random_state=42))
    stat0, p0 = shapiro(group0.sample(n0, random_state=42))

    print(f"정규성 p값 (group1): {p1:.4f}, (group0): {p0:.4f}")

    # 등분산성 검정
    stat, p = levene(group1, group0)
    print(f"등분산성 p값: {p:.4f}")

    # 장소 변수별 비모수 검정 
from scipy.stats import mannwhitneyu

for col in Location:
    group1 = df[df[TARGET]==1][col].dropna()
    group0 = df[df[TARGET]==0][col].dropna()

    stat, p = mannwhitneyu(group1, group0, alternative='two-sided')
    print(f"{col} - Mann-Whitney U p값: {p:.4f}")


    
#수치형 변수/ 이진형/ 범주형 각각 t검정, 비모수검정, 카이제곱 검정 

from scipy.stats import shapiro, ttest_ind, mannwhitneyu, chi2_contingency
import pingouin as pg   # 카이-제곱용

TARGET = 'host_is_superhost'

# 수치형 변수 리스트 (위도·경도·식별자 제외)
raw_num = [c for c in df.select_dtypes(include=['int64','float64']).columns
           if c not in ['latitude','longitude','host_id','id','host_is_superhost','Unnamed: 0']]

# 이진 수치형(0/1)만 골라내기
binary_num = [c for c in raw_num if df[c].dropna().isin([0,1]).all()]
continuous_num = [c for c in raw_num if c not in binary_num]

# 범주형 변수
cat_cols = df.select_dtypes(include=['object','category']).columns

results = []

# 연속형: 정규성 → t vs Mann-Whitney
def check_normality(series):
    return shapiro(series.dropna())[1] >= 0.05

for col in continuous_num:
    super = df[df[TARGET]==1][col].dropna()
    non   = df[df[TARGET]==0][col].dropna()
    
    if check_normality(super) and check_normality(non):
        stat, p = ttest_ind(super, non, equal_var=False)
        test = 't-test'
    else:
        stat, p = mannwhitneyu(super, non, alternative='two-sided')
        test = 'Mann-Whitney U'
    
    results.append({'variable':col, 'test':test, 'p':round(p,4)})

# 이진 수치형 & 범주형 → 카이제곱
for col in binary_num + cat_cols.tolist():
    ct = pd.crosstab(df[col], df[TARGET])
    chi2, p, _, _ = chi2_contingency(ct)
    results.append({'variable':col, 'test':'chi2', 'p':round(p,4)})

# 결과 정리
stat_df = pd.DataFrame(results).sort_values('p')
stat_df


# 연관성탐색 모델링 랜덤포레스트
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# 제외할 컬럼
exclude_cols = ['host_is_superhost', 'amenities', 'host_id', 'longitude', 'latitude','parsed_amenities']

# 설명 변수 설정 (원본 df에서 제외 컬럼 제외)
cols = [c for c in df.columns if c not in exclude_cols]
X = df[cols]

# 원핫인코딩 (범주형 변수 처리)
X = pd.get_dummies(X, drop_first=True)

# 타겟 변수
y = df['host_is_superhost']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 랜덤포레스트 모델 학습
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)

# 예측
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

# 평가 결과 출력
print("\n=== 테스트셋 평가 결과 ===")
print(classification_report(y_test, y_pred))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("AUC:", round(roc_auc_score(y_test, y_proba), 3))

# 변수 중요도 출력
importances = pd.Series(rf.feature_importances_, index=X.columns)
print("\n=== 변수 중요도 ===")
print(importances.sort_values(ascending=False).round(3))


# 변수중요도 기준으로 전략가능한 변수들 전략모델링 랜덤포레스트 
strategy_cols = ['amenities_cnt','availability_365','log_price','price','host_response_time_score','host_acceptance_rate_score',
                 'instant_bookable','host_about_length_group','room_type','neighbourhood_group_cleansed','host_has_profile_pic',
                 'neighborhood_overview_exists','name_length_group','description_length_group','is_long_term','accommodates','host_identity_verified',
                 'room_new_type','common_amenity_score', 'type_amenity_score']
# 중요 변수만 선택해서 전략 모델용 데이터셋 구성
X_top = df[strategy_cols]
X_top_encoded = pd.get_dummies(X_top)
# 학습/테스트 분할
X_train_top, X_test_top, y_train_top, y_test_top = train_test_split(
    X_top_encoded, y, test_size=0.2, random_state=42, stratify=y)

# 랜덤포레스트로 학습 (전략 모델)
rf_top = RandomForestClassifier(n_estimators=300, random_state=42)
rf_top.fit(X_train_top, y_train_top)
# 평가 지표
y_pred_top = rf_top.predict(X_test_top)
y_proba_top = rf_top.predict_proba(X_test_top)[:, 1]

print("\n=== 전략 모델 성능 평가  ===")
print(classification_report(y_test_top, y_pred_top))
print("AUC:", roc_auc_score(y_test_top, y_proba_top))


# 변수 중요도
importances2 = pd.Series(rf_top.feature_importances_, index=X_top_encoded.columns)
print("\n=== 변수 중요도 ===")
importances2.sort_values(ascending=False).round(2)


# 위치 'transport_count', 'infrastructure_count', 'tourism_count', 'poi_pca1' 변수들 랜덤포레스트 
Location = ['transport_count', 'infrastructure_count', 'tourism_count', 'poi_pca1']
TARGET = 'host_is_superhost'
from sklearn.ensemble import RandomForestClassifier

# X, y 분리
X = df[Location]
y = df[TARGET].astype(int)

rf = RandomForestClassifier(
    n_estimators=2000,  # 더 많은 트리
    max_depth=30,      # 최대 깊이 제한
    min_samples_split=15,  # 노드 분할 최소 샘플 수
    min_samples_leaf=10,    # 리프 노드 최소 샘플 수
    random_state=42,
    class_weight='balanced')

# 학습용/테스트용 데이터 나누기
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 모델 학습
rf.fit(X_train, y_train)

# 예측
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

# 평가
print("\n=== Random Forest ===")
print(classification_report(y_test, y_pred))
print("AUC:", round(roc_auc_score(y_test, y_prob), 4))
# 변수 중요도
importances_rf = pd.Series(rf.feature_importances_, index=Location)
print("\n=== Random Forest 변수 중요도 ===")
print(importances_rf.sort_values(ascending=False))




# 수치형데이터 중앙값 평균값 비교 
continuous_cols = [
    c for c in df.select_dtypes(include=['int64', 'float64']).columns
    if c not in [
        'host_is_superhost', # 종속변수
        'latitude', 'longitude', 'host_id', 'id', 'Unnamed: 0',
        # 이진 0/1 변수들 추가
        'host_identity_verified', 'host_location_boolean', 'host_location_ny',
        'neighborhood_overview_exists', 'is_long_term', 'instant_bookable',
        'is_activate', 'host_has_profile_pic','accommodates']]

# 중앙값 테이블
median_table = pd.DataFrame({
    'variable': continuous_cols,
    'superhost_median': [df[df['host_is_superhost'] == 1][col].median() for col in continuous_cols],
    'non_superhost_median': [df[df['host_is_superhost'] == 0][col].median() for col in continuous_cols]})

# 평균값 테이블 
avg_table = pd.DataFrame({
    'variable': continuous_cols,
    'superhost_avg': [df[df['host_is_superhost'] == 1][col].mean().round(2) for col in continuous_cols],
    'non_superhost_avg': [df[df['host_is_superhost'] == 0][col].mean().round(2) for col in continuous_cols]})

# 평균 + 중앙값 테이블 합치기
merged_table = pd.merge(avg_table,median_table,on='variable')

# 차이 컬럼 추가
merged_table['mean_diff'] = (merged_table['superhost_avg'] - merged_table['non_superhost_avg']).round(2)
merged_table['median_diff'] = (merged_table['superhost_median'] - merged_table['non_superhost_median']).round(2)

# 차이 기준 정렬 
merged_table.sort_values('mean_diff', ascending=False)


# 수치형 데이터(이진제외)시각화
import seaborn as sns
import matplotlib.pyplot as plt

continuous_cols = [
    'amenities_cnt', 'availability_365', 'price', 'log_price',
    'accommodates', 'host_acceptance_rate_score', 'host_response_time_score'
]

plt.figure(figsize=(14, 10))
for i, col in enumerate(continuous_cols, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x='host_is_superhost', y=col, data=df)
    plt.title(col)
    plt.xticks([0, 1], ['Not', 'Super'])
plt.tight_layout()
plt.show()


# 이진수치형 데이터 변수
bin_vars = ['host_location_boolean', 'host_location_ny',
        'neighborhood_overview_exists', 'is_long_term', 'instant_bookable',
        'is_activate', 'host_has_profile_pic'] 

bin_table = pd.DataFrame({'variable': bin_vars,
    'superhost_1(%)': [df[df['host_is_superhost']==1][col].mean().round(2)*100 for col in bin_vars],
    'superhost_0(%)': [(1-df[df['host_is_superhost']==1][col]).mean().round(2)*100 for col in bin_vars],
    'non_superhost_1(%)': [df[df['host_is_superhost']==0][col].mean().round(2)*100 for col in bin_vars],
    'non_superhost_0(%)': [(1-df[df['host_is_superhost']==0][col]).mean().round(2)*100 for col in bin_vars],
    'diff_1(%)': [df[df['host_is_superhost']==1][col].mean().round(2)*100 -
                 df[df['host_is_superhost']==0][col].mean().round(2)*100 for col in bin_vars]})

bin_table = bin_table.sort_values('diff_1(%)', ascending=False)
bin_table



# 범주형 변수 room_new_type
cat_var_room_new_type = 'room_new_type'
ct = pd.crosstab(df[cat_var_room_new_type], df['host_is_superhost'], normalize='columns') * 100
ct.round(1)

# 룸타입 
cat_var_room = 'room_type'
ct = pd.crosstab(df[cat_var_room], df['host_is_superhost'], normalize='columns') * 100
ct.round(1)
# 비슈퍼호스트일때와 슈퍼호스트일때 룸타입별 비율 

cat_var_name = 'name_length_group'
ct = pd.crosstab(df[cat_var_name], df['host_is_superhost'], normalize='columns') * 100
ct.round(1)

cat_var_description = 'description_length_group'
ct = pd.crosstab(df[cat_var_description], df['host_is_superhost'], normalize='columns') * 100
ct.round(1)

cat_var_hostabout = 'host_about_length_group'
ct = pd.crosstab(df[cat_var_hostabout], df['host_is_superhost'], normalize='columns') * 100
ct.round(1)

cat_var_structure = 'room_structure_type'
ct = pd.crosstab(df[cat_var_structure], df['host_is_superhost'], normalize='columns') * 100
ct.round(1)

cat_var_neighbourhood = 'neighbourhood_cleansed'
ct = pd.crosstab(df[cat_var_neighbourhood], df['host_is_superhost'], normalize='columns') * 100
ct.round(1)

cat_var_group = 'neighbourhood_group_cleansed'
ct = pd.crosstab(df[cat_var_group], df['host_is_superhost'], normalize='columns') * 100
ct.round(1)

# 범주형/ 이진형 데이터 시각화 
cat_cols = [
    'is_long_term', 'instant_bookable', 'neighborhood_overview_exists',
    'neighbourhood_group_cleansed', 'host_identity_verified',
    'room_type', 'host_has_profile_pic', 'room_new_type']

n_cols = 4
n_rows = (len(cat_cols) + n_cols - 1) // n_cols
plt.figure(figsize=(16, 3 * n_rows))

for i, col in enumerate(cat_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.countplot(x=col, hue='host_is_superhost', data=df)
    plt.title(col)
    plt.legend(title=None, labels=['Not', 'Super'])
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#Location 중앙값, 평균차

# 중앙값 테이블
Location_median = pd.DataFrame({
    'variable': Location,
    'superhost_median': [df[df['host_is_superhost'] == 1][col].median() for col in Location],
    'non_superhost_median': [df[df['host_is_superhost'] == 0][col].median() for col in Location]})

# 평균값 테이블 
Location_median_avg = pd.DataFrame({
    'variable': Location,
    'superhost_avg': [df[df['host_is_superhost'] == 1][col].mean().round(2) for col in Location],
    'non_superhost_avg': [df[df['host_is_superhost'] == 0][col].mean().round(2) for col in Location]})
 
# 평균 + 중앙값 테이블 합치기
Location_merged = pd.merge(Location_median_avg,Location_median,on='variable')

# 차이 컬럼 추가
Location_merged['mean_diff'] = (Location_merged['superhost_avg'] - Location_merged['non_superhost_avg']).round(2)
Location_merged['median_diff'] = (Location_merged['superhost_median'] - Location_merged['non_superhost_median']).round(2)

# 차이 기준 정렬
Location_merged.sort_values('mean_diff', ascending=False)

# 박스플롯 & 커널밀도 히스토그램 (각 변수별 슈퍼호스트 유무 분포)
import seaborn as sns
import matplotlib.pyplot as plt
location_vars = ['transport_count', 'infrastructure_count', 'tourism_count','poi_pca1']

for var in location_vars:
    plt.figure(figsize=(12, 5))

    # 박스플롯
    plt.subplot(1, 2, 1)
    sns.boxplot(x='host_is_superhost', y=var, data=df)
    plt.title(f'{var} by Superhost (Boxplot)')
    plt.xlabel('Superhost')
    plt.ylabel(var)

    # 히스토그램 + 커널밀도
    plt.subplot(1, 2, 2)
    sns.kdeplot(data=df[df['host_is_superhost'] == 1], x=var, label='Superhost', fill=True)
    sns.kdeplot(data=df[df['host_is_superhost'] == 0], x=var, label='Not Superhost', fill=True)
    plt.title(f'{var} Distribution by Superhost (KDE)')
    plt.xlabel(var)
    plt.legend()

    plt.tight_layout()
    plt.show()


# 슈퍼호스트 예측 모델링 로지스틱/랜덤포레스트 앙상블
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd

# 전략 변수 리스트
strategy_cols = [
    'amenities_cnt','availability_365','price','host_response_time_score',
    'host_acceptance_rate_score','instant_bookable','host_about_length_group','room_type',
    'neighbourhood_group_cleansed','host_has_profile_pic','neighborhood_overview_exists',
    'name_length_group','description_length_group','is_long_term','accommodates',
    'host_identity_verified','room_new_type','common_amenity_score', 'type_amenity_score']

# 설명 변수: 원핫 인코딩 포함
X_top = df[strategy_cols]
X_top_encoded = pd.get_dummies(X_top, drop_first=True)

# 목표 변수
y = df['host_is_superhost']

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_top_encoded, y, test_size=0.2, random_state=42, stratify=y)

# 개별 모델 정의
log_reg = LogisticRegression(max_iter=3000, random_state=42)
rf = RandomForestClassifier(n_estimators=300, random_state=42)

# 소프트 보팅 앙상블
ensemble = VotingClassifier(estimators=[('lr', log_reg), ('rf', rf)], voting='soft')
ensemble.fit(X_train, y_train)

# 예측 및 평가
y_pred = ensemble.predict(X_test)
y_proba = ensemble.predict_proba(X_test)[:, 1]

print("\n=== 소프트 보팅 앙상블 평가 결과 ===")
print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_proba))

# 로지스틱 회귀 계수 분석
log_reg.fit(X_train, y_train)
coeff_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': log_reg.coef_[0]}).sort_values(by='Coefficient', ascending=False)

print("\n=== 로지스틱 회귀 계수 상위 변수 ===")
print(coeff_df.round(3).head(10))

print("\n=== 로지스틱 회귀 계수 하위 변수 ===")
print(coeff_df.round(3).tail(10))


def preprocess_input(new_data_df, train_columns):
    """
    신규 데이터 전처리 함수
    - 원핫인코딩 적용
    - 훈련 데이터 컬럼과 일치하도록 맞춤
    """
    # 원핫인코딩 (drop_first=True 적용한 훈련과 동일하게)
    new_data_encoded = pd.get_dummies(new_data_df, drop_first=True)
    
    # 훈련 데이터 컬럼과 맞추기 (없는 컬럼은 0으로 채움)
    missing_cols = set(train_columns) - set(new_data_encoded.columns)
    for c in missing_cols:
        new_data_encoded[c] = 0
    
    # 순서 맞추기
    new_data_encoded = new_data_encoded[train_columns]
    
    return new_data_encoded

def predict_superhost(new_data_df, model, train_columns):
    """
    신규 데이터 받아서 슈퍼호스트 여부 예측
    """
    # 전처리
    X_new = preprocess_input(new_data_df, train_columns)
    
    # 예측 (확률)
    proba = model.predict_proba(X_new)[:,1]
    pred = model.predict(X_new)
    
    # 결과 반환
    result = new_data_df.copy()
    result['superhost_probability'] = proba
    result['superhost_prediction'] = pred
    
    return result

# 예측 돌리기 
'''
new_data_df = pd.DataFrame([{
    'amenities_cnt': 
    'availability_365': 
    'price': 
    'host_response_time_score': 
    'host_acceptance_rate_score': 
    'instant_bookable': 
    'host_about_length_group':            # ['short', 'med', 'long']
    'room_type':                          # ['Entire home/apt', 'Private room', 'Shared room', 'Hotel room']
    'neighbourhood_group_cleansed':       # ['Brooklyn', 'Manhattan', 'Queens', 'Bronx', 'Staten Island']
    'host_has_profile_pic': 
    'neighborhood_overview_exists':
    'name_length_group':                  # ['short', 'short_or_med', 'long']
    'description_length_group':           # ['short_or_avg', 'long']
    'is_long_term': 
    'accommodates': 
    'host_identity_verified': 
    'room_new_type':                      # ['low', 'low-mid', 'mid', 'high']
    'common_amenity_score': 
    'type_amenity_score': 
}])

# 예측 실행
result_df = predict_superhost(new_data_df, model=ensemble, train_columns=X_train.columns)

# 결과 보기
print(result_df[['superhost_probability', 'superhost_prediction']])
'''