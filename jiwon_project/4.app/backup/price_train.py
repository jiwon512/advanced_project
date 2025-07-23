import pandas as pd
import joblib
import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1) feature 리스트
cat_cols = [
    'nei_cluster_code', 'nei_borough',
    'room_type_code', 'room_group_code', 'room_structure', 'amen_grp',
    'info_des_len', 'info_name_len'
]
num_cols = [
    'latitude', 'longitude', 'accommodates', 'bath_score_mul',
    'amenities_cnt', 'review_scores_rating',
    'number_of_reviews', 'number_of_reviews_ltm', 'region_score_norm',
    'host_response_time_score', 'host_response_rate_score'
]
bin_cols = [
    'instant_bookable', 'is_long_term', 'host_is_superhost',
    'has_Air_conditioning', 'has_Wifi', 'has_Bathtub',
    'has_Carbon_monoxide_alarm', 'has_Elevator',
    'neighborhood_overview_exists'
]
other_flags = ['nei_cluster_grp01_high', 'nei_cluster_grp04_high']
features = cat_cols + num_cols + bin_cols + other_flags

# 2) 데이터 로드
DATA_PATH = "/Users/Jiwon/Documents/GitHub/advanced_project/hye_project/03_MachineLearning/for_machine_learning.csv"
df = pd.read_csv(DATA_PATH)

# 3) train/validation split
X = df[features]
y = df['log_price']
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2,
    stratify=df['room_type_ord'],
    random_state=42
)
print(f"Train samples: {len(X_tr)},  Validation samples: {len(X_val)}")

# 4) Optuna 스터디 로드 & 최적 파라미터 매핑
STUDY_PATH = 'optuna_study.pkl'  # 경로 필요시 수정
study = joblib.load(STUDY_PATH)
best = study.best_params.copy()
if 'lr' in best:   best['learning_rate']      = best.pop('lr')
if 'l2' in best:   best['l2_leaf_reg']        = best.pop('l2')
if 'bt' in best:   best['bagging_temperature']= best.pop('bt')
best_iter = best.get('iterations', best.get('cb_iter', 1000))

# 5) 전처리 정의 (HGB용)
preprocessor = ColumnTransformer([
    ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
], remainder='passthrough')

# 6-1) CatBoost 파이프라인
cat_pipeline = Pipeline([
    ('identity', FunctionTransformer()),
    ('cb', CatBoostRegressor(
        **best,
        iterations=best_iter,
        random_seed=42,
        verbose=False,
        cat_features=cat_cols + other_flags
    ))
])

# 6-2) HGB 파이프라인
hgb_pipeline = Pipeline([
    ('pre', preprocessor),
    ('hgb', HistGradientBoostingRegressor(
        learning_rate=best.get('hgb_lr', 0.05),
        max_leaf_nodes=best.get('hgb_leaves', 31),
        validation_fraction=0.1,
        n_iter_no_change=50,
        random_state=42
    ))
])

# 7) 스태킹 앙상블
stack = StackingRegressor(
    estimators=[('cat', cat_pipeline), ('hgb', hgb_pipeline)],
    final_estimator=RidgeCV(),
    cv=5, n_jobs=-1, passthrough=False
)

# 8) 학습 & 예측
stack.fit(X_tr, y_tr)
joblib.dump(stack, 'for_app.pkl')
print("Saved model to 'for_app.pkl'") 