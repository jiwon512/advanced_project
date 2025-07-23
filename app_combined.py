import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

# ====== 1. 경로/피처리스트/데이터 로드 ======
# 가격 예측 모델
PRICE_MODEL_PATH = "/Users/Jiwon/Documents/GitHub/advanced_project/jiwon_price.pkl"
PRICE_DF_PATH = "/Users/Jiwon/Documents/GitHub/advanced_project/jiwon_project/4.app/backup/processed_hye.csv"

# occupancy 예측 모델
OCC_MODEL_PATH = "/Users/Jiwon/Documents/GitHub/advanced_project/occupancy_voting_model.pkl"
OCC_DF_PATH = "/Users/Jiwon/Documents/GitHub/advanced_project/jiwon_project/presentation/jiwon_entire.csv"

# 가격 예측 피처리스트 (backup_app.py의 hye_features)
cat_cols = ['neigh_cluster_reduced','neighbourhood_group_cleansed','room_type_ord','room_new_type_ord','room_structure_type','amen_grp','description_length_group','name_length_group']
num_cols = ['latitude','longitude','accommodates','bath_score_mul','amenities_cnt','review_scores_rating','number_of_reviews','number_of_reviews_ltm','region_score_norm','host_response_time_score','host_response_rate_score']
bin_cols = ['instant_bookable','is_long_term','host_is_superhost','has_Air_conditioning','has_Wifi','has_Bathtub','has_Carbon_monoxide_alarm','has_Elevator','neighborhood_overview_exists']
other_flags = ['grp01_high','grp04_high']
hye_features = cat_cols + num_cols + bin_cols + other_flags

# occupancy 예측 피처리스트
occ_cols = [
 'host_is_superhost', 'host_total_listings_count', 'host_has_profile_pic', 'accommodates', 'beds',
 'availability_365', 'is_long_term', 'amenities_cnt', 'neighborhood_overview_exists',
 'name_length_group', 'description_length_group', 'host_about_length_group', 'host_location_ny',
 'is_private', 'bath_score_mul', 'is_activate', 'log_price', 'room_new_type_encoded',
 'neighbourhood_cluster', 'poi_pca', 'host_response_pca', 'host_verifications_count', 'score_info_pca'
]

# 데이터/모델 로드
@st.cache_data
def load_df(path):
    return pd.read_csv(path)
@st.cache_resource
def load_model(path):
    return joblib.load(path)
price_df = load_df(PRICE_DF_PATH)
occ_df = load_df(OCC_DF_PATH)
price_model = load_model(PRICE_MODEL_PATH)
occ_model = load_model(OCC_MODEL_PATH)

# ====== 2. 예측 함수 ======
def predict_price(row: dict) -> float:
    X = pd.DataFrame([row])[hye_features]
    return float(np.expm1(price_model.predict(X)[0]))

def predict_occupancy(row: dict) -> float:
    X = pd.DataFrame([row])[occ_cols].fillna(0)
    return float(occ_model.predict(X)[0])

# ====== 3. UI/UX ======
st.title("🗽 NYC Airbnb 호스트 전략 도우미 (통합)")
mode = st.radio("당신의 상태를 선택하세요", ["예비 호스트", "기존 호스트"])

# ====== 4. 예비 호스트 ======
if mode == "예비 호스트":
    st.header("🚀 희망 수입 달성을 위한 맞춤 준비 가이드")
    sel_boroughs = st.multiselect(
        "운영을 고려 중인 자치구(복수 선택 가능)",
        price_df['neighbourhood_group_cleansed'].unique().tolist(),
        default=["Manhattan"]
    )
    if not sel_boroughs:
        st.warning("최소 1개의 자치구를 선택해 주세요.")
        st.stop()
    accommodates = st.slider("최대 숙박 인원", 1, int(price_df['accommodates'].max()), 2)
    rt_ordinals  = sorted(price_df['room_type_ord'].unique())
    rt_labels    = ['Private room', 'Shared room', 'Entire home/apt', 'Hotel room']
    rt_choice_lb = st.selectbox("희망 룸 타입", rt_labels)
    rt_choice    = rt_ordinals[rt_labels.index(rt_choice_lb)]
    desired_month = st.number_input("희망 월수입 ($)", 0.0, 20000.0, 4000.0, 100.0)
    open_days = st.number_input("월 운영일 수", 1, 31, 30)
    target_price = desired_month / open_days
    st.markdown(f"➡️ **목표 1박 요금** : `${target_price:,.0f}`")

    if st.button("🔍 맞춤 추천 보기"):
        with st.spinner("⏳ 추천 계산 중…"):
            base_row = {**price_df[num_cols].median().to_dict(),
                        **price_df[cat_cols].mode().iloc[0].to_dict(),
                        **{c:0 for c in bin_cols},
                        **{f:0 for f in other_flags},
                        'accommodates': accommodates,
                        'room_type_ord': rt_choice,
                        'host_is_superhost': 0}
            recs = []
            for bor in sel_boroughs:
                for amen_grp in price_df['amen_grp'].unique():
                    for new_ord in price_df['room_new_type_ord'].unique():
                        row = base_row | {
                            'neighbourhood_group_cleansed': bor,
                            'amen_grp': amen_grp,
                            'room_new_type_ord': new_ord
                        }
                        # 가격 예측
                        price = predict_price(row)
                        # occupancy 예측용 입력값 생성 (공통 입력값만 반영, 나머지는 median/mode)
                        occ_defaults = occ_df[occ_cols].median(numeric_only=True).to_dict()
                        occ_defaults.update(occ_df[occ_cols].mode().iloc[0].to_dict())
                        occ_input = occ_defaults.copy()
                        occ_input['accommodates'] = accommodates
                        occ_input['host_is_superhost'] = 0
                        occ_input['room_new_type_encoded'] = row.get('room_new_type_ord', 0)
                        occ_input['amenities_cnt'] = row.get('amenities_cnt', occ_input.get('amenities_cnt', 0))
                        occ_days = predict_occupancy(occ_input)
                        annual_revenue = price * occ_days
                        recs.append({
                            '자치구': bor,
                            'Amenity 그룹': amen_grp,
                            '신규 룸그룹': new_ord,
                            '예측 1박 요금': f"${price:,.0f}",
                            '예상 연간 예약일수': f"{occ_days:,.0f}일",
                            '예상 연수익': f"${annual_revenue:,.0f}"
                        })
            rec_df = pd.DataFrame(recs)
            st.subheader("📋 추천 조합")
            st.table(rec_df)

# ====== 5. 기존 호스트 ======
if mode == "기존 호스트":
    st.header("📈 기존 호스트를 위한 전략 분석")
    st.info("(이 부분은 필요에 따라 추가 구현 가능)")
    st.write("가격예측, 예약일수예측, 연수익 계산 등 예비호스트와 동일하게 확장 가능") 