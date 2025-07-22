CSV_PATH = "/Users/Jiwon/Documents/GitHub/advanced_project/jiwon_project/csv_files/NY_Airbnb_original_df.csv"
print(f"▶ preprocessing.py 실행 시작, CSV_PATH={CSV_PATH}")


import numpy as np
import pandas as pd
import ast, re

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
# ColumnTransformer 는 여전히 compose 에 있음
from sklearn.compose import ColumnTransformer  
# SimpleImputer 은 impute 아래로 이동
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler,
    OneHotEncoder,
)
from sklearn.decomposition import PCA





# ----------------------------
# 1) 개별 커스텀 Transformer
# ----------------------------

class AmenitiesCounter(BaseEstimator, TransformerMixin):
    """리스트 형태 amenities → 길이 하나의 컬럼으로."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        # X: array-like of strings or lists
        return np.array([
            len(ast.literal_eval(x)) if isinstance(x, str) else len(x or [])
            for x in X.ravel()
        ])[:,None]

class VerificationsCounter(BaseEstimator, TransformerMixin):
    """host_verifications 컬럼(문자열 리스트) → 길이."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        cnts = []
        for x in X.ravel():
            if isinstance(x, list):      cnts.append(len(x))
            else:
                try:       cnts.append(len(ast.literal_eval(x)))
                except:   cnts.append(0)
        return np.array(cnts)[:,None]

class StructureCategoryMapper(BaseEstimator, TransformerMixin):
    """property_type/room_type → structure_category."""
    def __init__(self):
        self.residential = {
            'rental unit','home','condo','townhouse','cottage',
            'bungalow','villa','vacation home','earthen home',
            'ranch','casa particular','tiny home','entire home/apt'
        }
        self.apartment_suite = {
            'guest suite','loft','serviced apartment','aparthotel',
            'private room'
        }
        self.hotel_lodging = {
            'hotel','boutique hotel','bed and breakfast',
            'resort','hostel','guesthouse','hotel room'
        }
    def fit(self, X, y=None): return self
    def transform(self, X):
        out = []
        for pt, rt in X:
            pt_l = pt.strip().lower()
            rt_l = rt.strip().lower()
            if rt_l in self.residential or pt_l in self.residential:
                out.append('Residential')
            elif rt_l in self.apartment_suite or pt_l in self.apartment_suite:
                out.append('Apartment_Suite')
            elif rt_l in self.hotel_lodging or pt_l in self.hotel_lodging:
                out.append('Hotel_Lodging')
            else:
                out.append('Others')
        return np.array(out)[:,None]


def parse_baths(text):
    if pd.isna(text): return np.nan
    s = str(text).lower()
    m = re.search(r'(\d+(\.\d+)?)', s)
    if m: return float(m.group(1))
    if 'half' in s: return 0.5
    return np.nan

class BathroomProcessor(BaseEstimator, TransformerMixin):
    """bathrooms, bathrooms_text → bath_score_mul."""
    def __init__(self, w_private=1.0, w_shared=0.5):
        self.wp, self.ws = w_private, w_shared
    def fit(self, X, y=None): return self
    def transform(self, X):
        # X is array of shape (n,2): [bathrooms, bathrooms_text]
        baths, texts = X[:,0].astype(float), X[:,1]
        parsed = np.array([parse_baths(t) for t in texts])
        # where parsed notna replace baths
        mask = ~np.isnan(parsed)
        baths[mask] = parsed[mask]
        # treat zeros
        baths = np.where(baths==0, 1, baths)
        is_shared = np.char.find(texts.astype(str).astype(object), 'shared')>=0
        mul = np.where(~is_shared, self.wp, self.ws)
        bath_score = baths * mul
        bath_score = np.where(bath_score==0, 1, bath_score)
        return bath_score[:,None]


# ----------------------------
# 2) 컬럼 분류
# ----------------------------
numeric_features = [
    'price', 'amenities_cnt', 'minimum_nights', 'availability_365',
    'host_response_time_score','host_response_rate_score','host_acceptance_rate_score',
    'bath_score_mul',
    'number_of_reviews','number_of_reviews_ltm',
    'number_of_reviews_l30d','number_of_reviews_ly','reviews_per_month',
    'transport_count','infrastructure_count','tourism_count'
]
# 파생된 PCA 칼럼(나중에)
# poi_pca, host_response_pca, score_info_pca, review_info_pca
# → 여기서는 생략하고 pipeline 끝단에 추가할 수도 있음

categorical_features = [
    'instant_bookable','is_long_term','neighborhood_overview_exists',
    'name_length_group','description_length_group','host_about_length_group',
    'host_identity_verified','host_has_profile_pic','host_is_superhost',
    'host_location_boolean','host_location_ny',
    'is_private','is_activate'
]

# 멀티입력 컬럼 처리
amenities_feature        = ['amenities']
verifications_feature    = ['host_verifications']
structure_cat_input      = ['property_type','room_type']
bathroom_input           = ['bathrooms','bathrooms_text']

# ----------------------------
# 3) 파이프라인 정의
# ----------------------------
# 3‑1) 수치형 pipeline
num_pipeline = Pipeline([
    ('impute',  SimpleImputer(strategy='median')),
    ('scale',   StandardScaler()),
])

# 3‑2) 범주형 pipeline
cat_pipeline = Pipeline([
    ('impute',  SimpleImputer(strategy='most_frequent')),
    ('ohe',     OneHotEncoder(handle_unknown='ignore')),
    ('toarr',  FunctionTransformer(lambda X: X.toarray(), validate=False))
])

# 3‑3) amenities pipeline
amen_pipeline = Pipeline([
    ('cnt',     AmenitiesCounter()),
    ('scale',   StandardScaler()),
])

# 3‑4) verifications pipeline
verif_pipeline = Pipeline([
    ('cnt',     VerificationsCounter()),
    ('scale',   StandardScaler()),
])

# 3‑5) structure_category pipeline
struct_pipeline = Pipeline([
    ('map',     StructureCategoryMapper()),
    ('ohe',     OneHotEncoder(handle_unknown='ignore')),
    ('toarr',  FunctionTransformer(lambda X: X.toarray(), validate=False)),
])

# 3‑6) bathroom pipeline
bath_pipeline = Pipeline([
    ('proc',    BathroomProcessor()),
    ('scale',   StandardScaler()),
])

# 3‑7) ColumnTransformer 에 묶기
preprocessor = ColumnTransformer(transformers=[
    ('num',     num_pipeline,       numeric_features),
    ('cat',     cat_pipeline,       categorical_features),
    ('amen',    amen_pipeline,      amenities_feature),
    ('verif',   verif_pipeline,     verifications_feature),
    ('struct',  struct_pipeline,    structure_cat_input),
    ('bath',    bath_pipeline,      bathroom_input),
], remainder='drop')


# ----------------------------
# 4) PCA 예시 (poi_pca 하나만)
# ----------------------------
poi_cols = ['transport_count','infrastructure_count','tourism_count']
poi_pipeline = Pipeline([
    ('select',    FunctionTransformer(lambda df: df[poi_cols], validate=False)),
    ('scale',     StandardScaler()),
    ('pca',       PCA(n_components=1, random_state=42)),
])

# 전체 preprocessor + poi_pca 합치려면
full_preprocessor = ColumnTransformer(transformers=[
    ('base' , preprocessor,            numeric_features
                                     + categorical_features
                                     + amenities_feature
                                     + verifications_feature
                                     + structure_cat_input
                                     + bathroom_input),
    ('poi'  , poi_pipeline,           poi_cols),
    # 호스트 응답/리뷰/점수 PCA도 동일하게 추가 가능
], remainder='drop')


# ----------------------------
# 5) 사용 예시
# ----------------------------
if __name__=='__main__':
    df = pd.read_csv('/Users/Jiwon/Documents/GitHub/advanced_project/jiwon_project/csv_files/NY_Airbnb_original_df.csv')

    X = df.drop(columns=['estimated_occupancy_l365d'])
    y = df['estimated_occupancy_l365d']

    # fit_transform
    Xp = full_preprocessor.fit_transform(X)
    print("Transformed shape:", Xp.shape)
    # 저장
    import joblib
    joblib.dump(full_preprocessor, "models/preprocessor.joblib")
