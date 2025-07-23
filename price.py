import pandas as pd
import joblib
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# 1) feature 리스트 (실제 컬럼명에 맞게 수정)
cat_cols = [
    'neigh_cluster_reduced','neighbourhood_group_cleansed','room_type_ord','room_new_type_ord',
    'room_structure_type','amen_grp','description_length_group','name_length_group'
]
num_cols = [
    'latitude','longitude','accommodates','bath_score_mul','amenities_cnt','review_scores_rating',
    'number_of_reviews','number_of_reviews_ltm','region_score_norm','host_response_time_score','host_response_rate_score'
]
bin_cols = [
    'instant_bookable','is_long_term','host_is_superhost','has_Air_conditioning','has_Wifi',
    'has_Bathtub','has_Carbon_monoxide_alarm','has_Elevator','neighborhood_overview_exists'
]
other_flags = ['grp01_high','grp04_high']
features = cat_cols + num_cols + bin_cols + other_flags

DATA_PATH = "/Users/Jiwon/Documents/GitHub/advanced_project/jiwon_project/4.app/backup/processed_hye.csv"
df = pd.read_csv(DATA_PATH)

X = df[features]
y = df['log_price']
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=df['room_type_ord'], random_state=42
)

# CatBoost 파라미터 직접 지정
cat_params = {
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "random_seed": 42,
    "verbose": False,
    "cat_features": cat_cols + other_flags
}

cat_pipeline = Pipeline([
    ('identity', FunctionTransformer()),
    ('cb', CatBoostRegressor(**cat_params))
])

# HGB 파라미터 직접 지정
preprocessor = ColumnTransformer([
    ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
], remainder='passthrough')
hgb_pipeline = Pipeline([
    ('pre', preprocessor),
    ('hgb', HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_leaf_nodes=31,
        validation_fraction=0.1,
        n_iter_no_change=50,
        random_state=42
    ))
])

stack = StackingRegressor(
    estimators=[('cat', cat_pipeline), ('hgb', hgb_pipeline)],
    final_estimator=RidgeCV(),
    cv=5, n_jobs=-1, passthrough=False
)

stack.fit(X_tr, y_tr)
joblib.dump(stack, 'jiwon_price.pkl')
print("Saved model to 'jiwon_price.pkl'")