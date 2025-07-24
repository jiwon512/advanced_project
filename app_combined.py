import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost
import re

# ====== 1. 경로/피처리스트/데이터 로드 ======
# 가격 예측 모델
PRICE_MODEL_PATH = "/Users/hyeom/Documents/GitHub/advanced_project/jiwon_price.pkl"
PRICE_DF_PATH = "/Users/hyeom/Documents/GitHub/advanced_project/jiwon_project/4.app/backup/processed_hye.csv"

# occupancy 예측 모델
OCC_MODEL_PATH = "/Users/hyeom/Documents/GitHub/advanced_project/occupancy_voting_model.pkl"
OCC_DF_PATH = "/Users/hyeom/Documents/GitHub/advanced_project/jiwon_project/presentation/jiwon_entire.csv"

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
    import xgboost
    return pd.read_csv(path)
@st.cache_resource
def load_model(path):
    import xgboost
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
st.title("NYC Airbnb Host")
mode = st.radio("당신의 상태를 선택하세요", ["예비 호스트", "기존 호스트"])

# ====== 4. 예비 호스트 ======
if mode == "예비 호스트":
    st.header("Airbnb 호스트가 되어보세요")
    sel_boroughs = st.multiselect(
        "**Boroughs**",
        price_df['neighbourhood_group_cleansed'].unique().tolist(),
        default=["Manhattan"]
    )
    if not sel_boroughs:
        st.warning("최소 1개의 자치구를 선택해 주세요.")
        st.stop()
    # 동네 여러 개 선택 (플러스 버튼)
    for bor in sel_boroughs:
        neighs = price_df[price_df['neighbourhood_group_cleansed'] == bor]['neighbourhood_cleansed'].unique().tolist()
        if f'selected_neighs_{bor}' not in st.session_state:
            st.session_state[f'selected_neighs_{bor}'] = [neighs[0]]
        st.markdown(f"**{bor} town**")
        for i, n in enumerate(st.session_state[f'selected_neighs_{bor}']):
            col1, col2 = st.columns([4,1])
            with col1:
                st.session_state[f'selected_neighs_{bor}'][i] = st.selectbox(f"**town {i+1}**", neighs, index=neighs.index(n), key=f"neigh_{bor}_{i}")
            with col2:
                if st.button("➖", key=f"remove_{bor}_{i}"):
                    st.session_state[f'selected_neighs_{bor}'].pop(i)
                    st.experimental_rerun()
        if st.button(f"➕ {bor} 내 동네 추가", key=f"add_{bor}"):
            for n in neighs:
                if n not in st.session_state[f'selected_neighs_{bor}']:
                    st.session_state[f'selected_neighs_{bor}'].append(n)
                    break
            st.experimental_rerun()
    # 장기 렌트 (동네 선택 아래)
    is_long_term = st.toggle("장기 렌트")
    accommodates = st.slider("최대 숙박 인원", 1, int(price_df['accommodates'].max()), 2)
    rt_ordinals  = sorted(price_df['room_type_ord'].unique())
    rt_labels    = ['Private room', 'Shared room', 'Entire home/apt', 'Hotel room']
    rt_choice_lb = st.selectbox("희망 룸 타입", rt_labels)
    rt_choice    = rt_ordinals[rt_labels.index(rt_choice_lb)]

    if st.button("🔍 맞춤 추천 보기"):
        with st.spinner("⏳ 추천 계산 중…"):
            base_row = {**price_df[num_cols].median().to_dict(),
                        **price_df[cat_cols].mode().iloc[0].to_dict(),
                        **{c:0 for c in bin_cols},
                        **{f:0 for f in other_flags},
                        'accommodates': accommodates,
                        'room_type_ord': rt_choice,
                        'host_is_superhost': 0,
                        'is_long_term': int(is_long_term)}
            recs = []
            for bor in sel_boroughs:
                for neigh in st.session_state[f'selected_neighs_{bor}']:
                    for amen_grp in price_df['amen_grp'].unique():
                        for new_ord in price_df['room_new_type_ord'].unique():
                            row = base_row | {
                                'neighbourhood_group_cleansed': bor,
                                'neighbourhood_cleansed': neigh,
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
                            occ_input['neighbourhood_group_cleansed'] = bor
                            occ_input['neighbourhood_cleansed'] = neigh
                            occ_input['is_long_term'] = int(is_long_term)
                            occ_days = predict_occupancy(occ_input)
                            annual_revenue = price * occ_days
                            recs.append({
                                '자치구': bor,
                                '동네': neigh,
                                'Amenity 그룹': amen_grp,
                                '신규 룸그룹': new_ord,
                                '예측 1박 요금': price,
                                '예상 연간 예약일수': occ_days,
                                '예상 연수익': annual_revenue
                            })
            rec_df = pd.DataFrame(recs)
            # 동네별 멘트+추천조합 버튼+표
            for bor in sel_boroughs:
                for neigh in st.session_state[f'selected_neighs_{bor}']:
                    # 슈퍼호스트+is_activate만 필터
                    df_super = occ_df[
                        (occ_df['neighbourhood_group_cleansed'] == bor) &
                        (occ_df['neighbourhood_cleansed'] == neigh) &
                        (occ_df['host_is_superhost'] == 1) &
                        (occ_df['is_activate'] == 1)
                    ]
                    avg_price = np.mean(np.expm1(df_super['log_price'])) if len(df_super) > 0 else 0
                    avg_occ = np.mean(df_super['estimated_occupancy_l365d']) if len(df_super) > 0 else 0
                    avg_revenue = avg_price * avg_occ
                    st.markdown(
                        f"""<div style='font-size:1.3em; font-weight:bold; color:#222; text-align:center; margin-bottom:10px;'>
                        {bor}, {neigh}에서<br>
                        연간 <span style='color:#FF5A5F'>{avg_revenue:,.0f}달러</span>를 벌어보세요!
                        </div>""", unsafe_allow_html=True
                    )
                    # 표 바로 아래에, 인덱스 없이, 스크롤 없이, 가운데 정렬
                    df_show = df_super.copy()
                    if 'log_price' in df_show.columns:
                        df_show['1박당 가격'] = np.expm1(df_show['log_price']).round(0).astype(int)
                    if 'estimated_occupancy_l365d' in df_show.columns and '1박당 가격' in df_show.columns:
                        df_show['연간수익'] = (df_show['1박당 가격'] * df_show['estimated_occupancy_l365d']).round(0).astype(int)
                    show_cols = ['1박당 가격', 'beds', 'amenities_cnt', 'tourism_count', 'infrastructure_count', 'amenity_group', '연간수익']
                    show_cols = [col for col in show_cols if col in df_show.columns]
                    if show_cols:
                        # 연간수익 기준 내림차순 정렬 후 상위 10개만
                        df_show = df_show.sort_values('연간수익', ascending=False).head(10)
                        st.markdown("<div style='display:flex; justify-content:center;'>", unsafe_allow_html=True)
                        st.table(df_show[show_cols].reset_index(drop=True))
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.info("해당 동네의 슈퍼호스트 숙소 정보가 없습니다.")


# ====== 5. 기존 호스트 ======
if mode == "기존 호스트":
    st.header("📈 기존 호스트를 위한 전략 분석")
    st.info("(이 부분은 필요에 따라 추가 구현 가능)")
    st.write("가격예측, 예약일수예측, 연수익 계산 등 예비호스트와 동일하게 확장 가능") 