import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
import joblib

# 데이터 불러오기
df = pd.read_csv("/Users/Jiwon/Documents/GitHub/advanced_project/jiwon_project/presentation/jiwon_entire.csv")
occ_cols = [
    'host_is_superhost', 'host_total_listings_count', 'host_has_profile_pic', 'accommodates', 'beds',
    'availability_365', 'is_long_term', 'amenities_cnt', 'neighborhood_overview_exists',
    'name_length_group', 'description_length_group', 'host_about_length_group', 'host_location_ny',
    'is_private', 'bath_score_mul', 'is_activate', 'log_price', 'room_new_type_encoded',
    'neighbourhood_cluster', 'poi_pca', 'host_response_pca', 'host_verifications_count', 'score_info_pca'
]
TARGET = "estimated_occupancy_l365d"
X = df[occ_cols].fillna(0)
y = df[TARGET]

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 정의 (경량화)
rf = RandomForestRegressor(n_estimators=50, max_depth=7, random_state=42)
xgb = XGBRegressor(objective="reg:squarederror", n_estimators=50, max_depth=5, learning_rate=0.05, random_state=42)

# 앙상블
voting = VotingRegressor([("rf", rf), ("xgb", xgb)])
voting.fit(X_train, y_train)

# 모델 저장 (이 환경에서!)
joblib.dump(voting, "occupancy_voting_model.pkl", compress=3)