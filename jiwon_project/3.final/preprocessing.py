# preprocessing.py
import os
import ast, re

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from sklearn.base      import BaseEstimator, TransformerMixin
from sklearn.compose   import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute    import SimpleImputer
from sklearn.pipeline  import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.cluster   import KMeans
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# 설정: 파일 경로
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH     = "/Users/Jiwon/Documents/GitHub/advanced_project/jiwon_project/csv_files/NY_Airbnb_original_df.csv"
OUTPUT_DIR   = "models"
PREP_PATH    = os.path.join(OUTPUT_DIR, "preprocessor.joblib")

# ─────────────────────────────────────────────────────────────────────────────
# 0) 원본 읽기
# ─────────────────────────────────────────────────────────────────────────────
print(f"▶ preprocessing.py 시작: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# 1) room_new_type 군집화 (원본 df 에 컬럼 추가)
# ─────────────────────────────────────────────────────────────────────────────
# (예시: OSMnx API 호출 부분 생략하고 'room_structure_type' 자체에 기반)
# --- 1-1) 원래 있던 room_structure_type 전처리 가정
df['room_structure_type'] = df['room_structure_type'].fillna('others').str.lower().str.strip()
# --- 1-2) 그룹별 log(price) 통계 집계
grp = df.groupby('room_structure_type')['price'].median().reset_index()
# --- 1-3) KMeans 로 4개 군집
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler as SS
Xg = SS().fit_transform(np.log1p(grp[['price']]))
km = KMeans(n_clusters=4, random_state=42).fit(Xg)
grp['cluster'] = km.labels_
# --- 1-4) 라벨 이름 매핑
name_map = {i:f"group_{i}" for i in grp['cluster'].unique()}
df = df.merge(grp[['room_structure_type','cluster']], on='room_structure_type', how='left')
df['room_new_type'] = df['cluster'].map(name_map)
df.drop(columns=['cluster'], inplace=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2) 이상치 제거 (IQR*1.5)
# ─────────────────────────────────────────────────────────────────────────────
def iqr_bounds(s, factor=1.5):
    q1, q3 = np.percentile(s.dropna(), [25,75])
    iqr = q3 - q1
    return q1 - factor*iqr, q3 + factor*iqr

# price 기준으로 room_new_type별 outlier 마스크
masks = []
for grp_name, sub in df.groupby('room_new_type'):
    lo, hi = iqr_bounds(np.log1p(sub['price']), factor=1.5)
    masks.append(sub.index[(np.log1p(sub['price']) >= lo) & (np.log1p(sub['price']) <= hi)])
good_idx = pd.Index(np.concatenate(masks))
df = df.loc[good_idx].reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3) custom Transformer 정의
# ─────────────────────────────────────────────────────────────────────────────
class AmenitiesCounter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return np.array([
            len(ast.literal_eval(x)) if isinstance(x,str) else len(x or [])
            for x in X.ravel()
        ])[:,None]

class VerificationsCounter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        out=[]
        for x in X.ravel():
            if isinstance(x,list): out.append(len(x))
            else:
                try: out.append(len(ast.literal_eval(x)))
                except: out.append(0)
        return np.array(out)[:,None]

def parse_bath(txt):
    if pd.isna(txt): return np.nan
    m=re.search(r"(\d+(\.\d+)?)", str(txt).lower())
    if m: return float(m.group(1))
    return 0.5 if 'half' in str(txt).lower() else np.nan

class BathScore(BaseEstimator, TransformerMixin):
    def fit(self,X,y=None): return self
    def transform(self, X):
        baths = X[:,0].astype(float)
        txts  = X[:,1]
        parsed = np.array([parse_bath(t) for t in txts])
        mask   = ~np.isnan(parsed)
        baths[mask]=parsed[mask]
        baths = np.where(baths==0,1,baths)
        shared = np.char.find(txts.astype(str),'shared')>=0
        mult   = np.where(shared,0.5,1.0)
        score  = baths*mult
        return np.where(score==0,1,score)[:,None]

# ─────────────────────────────────────────────────────────────────────────────
# 4) 파이프라인용 컬럼 분류
# ─────────────────────────────────────────────────────────────────────────────
num_feats = [
  'price','amenities_cnt','minimum_nights','availability_365',
  'host_response_time_score','host_response_rate_score','host_acceptance_rate_score',
  'bath_score_mul',
  'number_of_reviews','number_of_reviews_ltm','number_of_reviews_l30d','number_of_reviews_ly','reviews_per_month',
  'transport_count','infrastructure_count','tourism_count'
]
cat_feats = [
  'instant_bookable','is_long_term','neighborhood_overview_exists',
  'name_length_group','description_length_group','host_about_length_group',
  'host_identity_verified','host_has_profile_pic','host_is_superhost',
  'host_location_boolean','host_location_ny',
  'is_private','is_activate','room_new_type'
]
amen_feats  = ['amenities']
verif_feats = ['host_verifications']
bath_feats  = ['bathrooms','bathrooms_text']

# ─────────────────────────────────────────────────────────────────────────────
# 5) ColumnTransformer & Pipeline 정의
# ─────────────────────────────────────────────────────────────────────────────
num_pipe = Pipeline([
    ('imp',  SimpleImputer(strategy='median')),
    ('sc',   StandardScaler()),
])
cat_pipe = Pipeline([
    ('imp',  SimpleImputer(strategy='most_frequent')),
    ('ohe',  OneHotEncoder(handle_unknown='ignore')),          # sparse=False 는 최신버전 기본
    ('arr',  FunctionTransformer(lambda X: X.toarray(), validate=False)),
])
amen_pipe = Pipeline([
    ('cnt', AmenitiesCounter()),
    ('sc',  StandardScaler()),
])
verif_pipe = Pipeline([
    ('cnt', VerificationsCounter()),
    ('sc',  StandardScaler()),
])
bath_pipe = Pipeline([
    ('proc', BathScore()),
    ('sc',   StandardScaler()),
])

base_pre   = ColumnTransformer([
    ('num',    num_pipe,    num_feats),
    ('cat',    cat_pipe,    cat_feats),
    ('amen',   amen_pipe,   amen_feats),
    ('verif',  verif_pipe,  verif_feats),
    ('bath',   bath_pipe,   bath_feats),
], remainder='drop')

# ─────────────────────────────────────────────────────────────────────────────
# 6) 추가 PCA pipeline (poi, host_res, score, review)
# ─────────────────────────────────────────────────────────────────────────────
def make_pca(cols):
    return Pipeline([
        ('sel', FunctionTransformer(lambda df: df[cols], validate=False)),
        ('sc',  StandardScaler()),
        ('pc',  PCA(n_components=1, random_state=42)),
    ])

poi_cols   = ['transport_count','infrastructure_count','tourism_count']
resp_cols  = ['host_response_time_score','host_response_rate_score','host_acceptance_rate_score']
score_cols = ['review_scores_rating','review_scores_accuracy','review_scores_cleanliness',
              'review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value']
rev_cols   = ['number_of_reviews','number_of_reviews_ltm','number_of_reviews_l30d','number_of_reviews_ly','reviews_per_month']

pca_pre = ColumnTransformer([
    ('poi',   make_pca(poi_cols),   poi_cols),
    ('resp',  make_pca(resp_cols),  resp_cols),
    ('score', make_pca(score_cols), score_cols),
    ('rev',   make_pca(rev_cols),   rev_cols),
], remainder='drop')

# ─────────────────────────────────────────────────────────────────────────────
# 7) 전체 합친 full_preprocessor
# ─────────────────────────────────────────────────────────────────────────────
full_preprocessor = ColumnTransformer([
    ('base', base_pre,    num_feats + cat_feats + amen_feats + verif_feats + bath_feats),
    ('pca',  pca_pre,      poi_cols + resp_cols + score_cols + rev_cols)
], remainder='drop')

# ─────────────────────────────────────────────────────────────────────────────
# 8) fit & dump
# ─────────────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    print(" · fitting preprocessor…")
    # X, y 분리
    X = df.drop(columns=['estimated_occupancy_l365d'])
    y = df['estimated_occupancy_l365d']
    # fit
    full_preprocessor.fit(X, y)
    # 저장
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(full_preprocessor, PREP_PATH)
    print(f" · 저장 완료: {PREP_PATH}")
