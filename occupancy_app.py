import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ====== 모델/데이터 경로 ======
OCC_MODEL_PATH = "/Users/Jiwon/Documents/GitHub/advanced_project/occupancy_voting_model.pkl"
OCC_DF_PATH = "/Users/Jiwon/Documents/GitHub/advanced_project/jiwon_project/presentation/jiwon_entire.csv"

# ====== 피처리스트 ======
occ_cols = [
    'host_is_superhost', 'host_total_listings_count', 'host_has_profile_pic', 'accommodates', 'beds',
    'availability_365', 'is_long_term', 'amenities_cnt', 'neighborhood_overview_exists',
    'name_length_group', 'description_length_group', 'host_about_length_group', 'host_location_ny',
    'is_private', 'bath_score_mul', 'is_activate', 'log_price', 'room_new_type_encoded',
    'neighbourhood_cluster', 'poi_pca', 'host_response_pca', 'host_verifications_count', 'score_info_pca'
]

# ====== 데이터/모델 로드 ======
@st.cache_data
def load_df(path):
    return pd.read_csv(path)
@st.cache_resource
def load_model(path):
    return joblib.load(path)

occ_df = load_df(OCC_DF_PATH)
occ_model = load_model(OCC_MODEL_PATH)

# ====== 예측 함수 ======
def predict_occupancy(row: dict) -> float:
    X = pd.DataFrame([row])[occ_cols].fillna(0)
    return float(occ_model.predict(X)[0])

# ====== Streamlit UI ======
st.title("🔢 연간 예약일수(occupancy days) 예측기")

# 주요 입력값만 받기 (나머지는 median/mode로 자동 채움)
accommodates = st.slider("최대 숙박 인원", 1, int(occ_df['accommodates'].max()), 2)
beds = st.slider("침대 개수", 1, int(occ_df['beds'].max()), 1)
host_is_superhost = st.selectbox("슈퍼호스트 여부", [0, 1], format_func=lambda x: "Yes" if x else "No")
amenities_cnt = st.slider("어매니티 개수", 0, int(occ_df['amenities_cnt'].max()), 5)

if st.button("연간 예약일수 예측"):
    # 기본값 세팅
    defaults = occ_df[occ_cols].median(numeric_only=True).to_dict()
    defaults.update(occ_df[occ_cols].mode().iloc[0].to_dict())
    # 입력값 반영
    defaults['accommodates'] = accommodates
    defaults['beds'] = beds
    defaults['host_is_superhost'] = host_is_superhost
    defaults['amenities_cnt'] = amenities_cnt
    # 예측
    occ_days = predict_occupancy(defaults)
    st.success(f"예상 연간 예약일수: {occ_days:,.0f}일") 