기존에 short_or_mid나 short_or_avg로 묶었던 길이 그룹을 단순히 **'short'**으로 통일해달라는 말씀이시죠? 네, 그렇게 수정하여 코드를 다시 제공해 드릴게요.

Streamlit 앱 코드 (수정됨: 길이 그룹 'short'으로 통일)
아래 코드에서 get_name_length_group, get_description_length_group, get_host_about_length_group 함수가 변경되었습니다. 이제 각 길이 그룹 함수의 반환 값 중 'short_or_mid'와 'short_or_avg'는 모두 **'short'**으로 변경됩니다.

Python

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

# --- 1. 데이터 및 모델 로드 ---
@st.cache_data
def load_pipeline(path):
    """지정된 경로에서 모델 파이프라인을 로드합니다."""
    try:
        pipeline = joblib.load(path)
        return pipeline
    except Exception as e:
        st.error(f"모델 파이프라인 로드 실패: {e}")
        return None

# 모델 파일 경로 (실제 경로에 맞게 수정해주세요)
MODEL_PATH = 'superhost_pipeline_rf.pkl'
pipeline = load_pipeline(MODEL_PATH)

if pipeline is None:
    st.stop() # 모델 로드 실패 시 앱 중단

# --- 2. 원시 데이터를 모델 입력 피처로 변환하는 유틸리티 함수 ---

# 1. host_response_time → 점수 변환 함수
def response_time_to_score(response_time_str):
    mapping = {
        'within an hour': 4,
        'within a few hours': 3,
        'within a day': 2,
        'a few days or more': 1
    }
    return mapping.get(response_time_str.lower(), 0)

# 2. host_acceptance_rate(0~100) → 점수 변환 함수
def acceptance_rate_to_score(rate_percent):
    if pd.isna(rate_percent) or rate_percent < 0 or rate_percent > 100:
        return 0
    rate = rate_percent / 100
    if rate <= 0.25:
        return 1
    elif rate <= 0.5:
        return 2
    elif rate <= 0.75:
        return 3
    else:
        return 4

# 3. amenities 점수 계산 함수
common_amenities = ['Carbon monoxide alarm', 'Essentials', 'Hangers', 'Smoke alarm', 'Wifi']

type_amenity_dict = {
    'high': ['Air conditioning', 'Building staff', 'Elevator', 'Gym', 'Heating', 'Paid parking off premises', 'Shampoo'],
    'low-mid': ['Cleaning products', 'Dining table', 'Exterior security cameras on property', 'Free street parking',
                'Freezer', 'Laundromat nearby', 'Lock on bedroom door', 'Microwave'],
    'mid': ['Cooking basics', 'Kitchen', 'Oven'],
    'upper-mid': ['Bathtub', 'Cleaning products', 'Cooking basics', 'Dishes and silverware', 'Elevator', 'Freezer']
}

def calc_amenity_scores(amenities_list, room_new_type):
    if not amenities_list:
        return 0.0, 0.0

    cleaned_amenities = [re.sub(r'[\uD800-\uDFFF]', '', a).strip().lower() for a in amenities_list]

    cleaned_common_amenities = [re.sub(r'[\uD800-\uDFFF]', '', a).strip().lower() for a in common_amenities]
    common_match_count = sum(1 for a in cleaned_amenities if a in cleaned_common_amenities)
    common_score = common_match_count / len(cleaned_common_amenities) if cleaned_common_amenities else 0

    type_amenities = type_amenity_dict.get(room_new_type, [])
    cleaned_type_amenities = [re.sub(r'[\uD800-\uDFFF]', '', a).strip().lower() for a in type_amenities]
    type_match_count = sum(1 for a in cleaned_amenities if a in cleaned_type_amenities)
    type_score = type_match_count / len(cleaned_type_amenities) if cleaned_type_amenities else 0

    return round(common_score, 3), round(type_score, 3)

# 4. 길이 그룹화 함수들 (수정됨: 'short'으로 통일)
def get_name_length_group(length):
    """숙소 이름 길이를 그룹화합니다. (mid: 38)"""
    if length == 0:
        return '없음'
    elif length > 38:
        return 'long'
    else:
        return 'short' # 'short_or_mid' -> 'short'

def get_description_length_group(length):
    """숙소 상세 설명 길이를 그룹화합니다. (avg: 359)"""
    if length == 0:
        return '없음'
    elif length > 359:
        return 'long'
    else:
        return 'short' # 'short_or_avg' -> 'short'

def get_host_about_length_group(length):
    """호스트 소개글 길이를 그룹화합니다. (mid: 81)"""
    if length == 0:
        return '없음'
    elif length > 81:
        return 'long'
    else:
        return 'short' # 'short_or_mid' -> 'short'

# --- 3. 예측 함수 정의 ---
def predict_superhost(input_data_dict: dict, pipeline) -> tuple:
    """
    단일 입력 딕셔너리를 받아 슈퍼호스트 여부를 예측하고 확률을 반환합니다.
    주의: input_data_dict는 모델 학습 시 사용된 strategy_cols와 동일한 키를 포함해야 합니다.
    """
    strategy_cols = [
        'amenities_cnt', 'availability_365', 'price', 'host_about_length_group',
        'room_type', 'name_length_group', 'description_length_group',
        'host_has_profile_pic', 'host_response_time_score', 'type_amenity_score',
        'common_amenity_score', 'host_acceptance_rate_score',
        'host_identity_verified', 'is_long_term', 'accommodates'
    ]

    try:
        X_new = pd.DataFrame([input_data_dict])[strategy_cols]
    except KeyError as e:
        st.error(f"입력 데이터에 필수 컬럼이 누락되었습니다: {e}. 모든 'strategy_cols'를 포함해야 합니다.")
        return None, None

    pred = pipeline.predict(X_new)[0]
    proba = pipeline.predict_proba(X_new)[0, 1] # 슈퍼호스트(클래스 1)일 확률

    return pred, proba

# --- 4. Streamlit 앱 UI 구성 ---
st.set_page_config(layout="wide")
st.title("🌟 Airbnb 슈퍼호스트 예측 도우미")

st.markdown("""
이 앱은 입력된 숙소 정보를 바탕으로 해당 숙소가 슈퍼호스트의 조건을 만족할 가능성을 예측합니다.
""")

st.subheader("🏡 숙소 정보 입력")

col1, col2 = st.columns(2)

with col1:
    st.markdown("##### 기본 정보")
    amenities_cnt = st.number_input("편의시설 개수", min_value=0, max_value=50, value=15)
    availability_365 = st.slider("1년 중 예약 가능 일수", min_value=0, max_value=365, value=180)
    price = st.number_input("1박 요금 ($)", min_value=10, max_value=1000, value=100)
    accommodates = st.number_input("최대 숙박 인원", min_value=1, max_value=16, value=2)

    st.markdown("##### 호스트 정보")
    host_has_profile_pic = st.selectbox("호스트 프로필 사진 유무", [True, False], format_func=lambda x: "있음" if x else "없음")
    host_identity_verified = st.selectbox("호스트 신원 인증 여부", [True, False], format_func=lambda x: "인증됨" if x else "미인증")
    host_response_time_raw = st.selectbox(
        "호스트 응답 시간",
        ['within an hour', 'within a few hours', 'within a day', 'a few days or more', 'N/A']
    )
    host_acceptance_rate_raw = st.slider("호스트 수락률 (%)", min_value=0, max_value=100, value=85)

with col2:
    st.markdown("##### 숙소 특징")
    room_type = st.selectbox("룸 타입", ['Entire home/apt', 'Private room', 'Shared room', 'Hotel room'])
    is_long_term = st.selectbox("장기 숙박 허용 여부", [True, False], format_func=lambda x: "허용" if x else "불허")

    # 길이 그룹화를 위한 원시 길이 입력 필드 추가
    st.markdown("##### 길이 정보")
    host_about_length_input = st.number_input("호스트 소개글 길이 (글자 수)", min_value=0, value=100)
    name_length_input = st.number_input("숙소 이름 길이 (글자 수)", min_value=0, value=20)
    description_length_input = st.number_input("숙소 상세 설명 길이 (글자 수)", min_value=0, value=500)

    st.markdown("##### 편의시설 정보")
    all_amenities_options = [
        'Wifi', 'Essentials', 'Hangers', 'Smoke alarm', 'Carbon monoxide alarm', 'Air conditioning',
        'Heating', 'Kitchen', 'Oven', 'Microwave', 'Shampoo', 'Bathtub', 'Elevator', 'Gym',
        'Free parking', 'Paid parking off premises', 'Cleaning products', 'Dining table',
        'Exterior security cameras on property', 'Freezer', 'Laundromat nearby', 'Lock on bedroom door',
        'Cooking basics', 'Dishes and silverware', 'Building staff'
    ]
    selected_amenities_raw = st.multiselect("주요 편의시설 선택 (슈퍼호스트 관련 편의시설 기준)", all_amenities_options,
                                            default=['Wifi', 'Essentials', 'Hangers', 'Smoke alarm', 'Kitchen', 'Oven'])
    room_new_type_for_amenity_score = st.selectbox(
        "숙소 타입 (편의시설 점수 계산용)", ['mid', 'high', 'low-mid', 'upper-mid']
    )


# 예측 버튼
if st.button("슈퍼호스트 가능성 예측하기"):
    # 1. 원시 입력 데이터를 점수화 및 가공
    host_response_time_score = response_time_to_score(host_response_time_raw)
    host_acceptance_rate_score = acceptance_rate_to_score(host_acceptance_rate_raw)
    common_amenity_score, type_amenity_score = calc_amenity_scores(
        selected_amenities_raw, room_new_type_for_amenity_score
    )

    # 새로 정의한 길이 그룹화 함수 적용 (이제 'short'으로 통일)
    host_about_length_group = get_host_about_length_group(host_about_length_input)
    name_length_group = get_name_length_group(name_length_input)
    description_length_group = get_description_length_group(description_length_input)


    # 2. 모델 예측에 필요한 최종 딕셔너리 구성
    input_for_prediction = {
        'amenities_cnt': amenities_cnt,
        'availability_365': availability_365,
        'price': price,
        'host_about_length_group': host_about_length_group,
        'room_type': room_type,
        'name_length_group': name_length_group,
        'description_length_group': description_length_group,
        'host_has_profile_pic': host_has_profile_pic,
        'host_response_time_score': host_response_time_score,
        'type_amenity_score': type_amenity_score,
        'common_amenity_score': common_amenity_score,
        'host_acceptance_rate_score': host_acceptance_rate_score,
        'host_identity_verified': host_identity_verified,
        'is_long_term': is_long_term,
        'accommodates': accommodates
    }

    # 3. 예측 함수 호출
    prediction, probability = predict_superhost(input_for_prediction, pipeline)

    # 4. 결과 표시
    st.subheader("📊 예측 결과")
    if prediction is not None:
        if prediction == 1:
            st.success(f"이 숙소는 슈퍼호스트가 될 가능성이 높습니다! (확률: **{probability:.2%}**)")
        else:
            st.info(f"이 숙소는 현재 슈퍼호스트가 아닐 가능성이 높습니다. (확률: **{1-probability:.2%}**)")

        st.progress(probability, text=f"슈퍼호스트가 될 확률: {probability:.2%}")

        st.markdown("""
        ---
        **참고:** 이 예측은 입력된 정보와 학습된 모델을 기반으로 합니다. 실제 결과는 다를 수 있습니다.
        """)