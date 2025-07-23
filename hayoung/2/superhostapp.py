import streamlit as st
import pandas as pd
import numpy as np
import joblib
import ast

# ────────────────────────────────────────────────
# 1) 모델 로드
@st.cache_resource
def load_pipeline(path):
    return joblib.load(path)

MODEL_PATH = "superhost_pipeline_rf.pkl"  # pkl 파일 경로
pipeline = load_pipeline(MODEL_PATH)

# ────────────────────────────────────────────────
# 2) 점수 변환 함수
def convert_response_time(response_time):
    mapping = {
        '1시간 이내': 4,
        '몇 시간 이내': 3,
        '하루 이내': 2,
        '며칠 이내': 1
    }
    return mapping.get(response_time.strip(), 0)

def convert_acceptance_rate(rate_str):
    try:
        rate = float(rate_str.strip('%'))
    except:
        return 0
    if rate >= 95:
        return 4
    elif rate >= 90:
        return 3
    elif rate >= 80:
        return 2
    elif rate > 0:
        return 1
    else:
        return 0

def convert_amenities(amenities_str, keywords):
    try:
        amenities = ast.literal_eval(amenities_str)
    except:
        return 0.0
    match_count = sum(1 for item in amenities if any(kw in item.lower() for kw in keywords))
    return round(match_count / len(keywords), 2)

def convert_profile_pic(value):
    return 1 if value == "있음" else 0

def convert_identity_verified(value):
    return 1 if value == "있음" else 0

def convert_long_term(value):
    return 1 if value == "장기" else 0

# ────────────────────────────────────────────────
# 3) Streamlit UI
st.title("🏠 슈퍼호스트 예측기")

st.subheader("📋 기본 정보 입력")
col1, col2 = st.columns(2)
with col1:
    host_response_time = st.selectbox("호스트 응답 시간", ["1시간 이내", "몇 시간 이내", "하루 이내", "며칠 이내"])
    host_acceptance_rate = st.text_input("호스트 수락률 (예: 98%)")
    host_has_profile_pic = st.selectbox("프로필 사진 여부", ["있음", "없음"])
    host_identity_verified = st.selectbox("신분 인증 여부", ["있음", "없음"])
with col2:
    is_long_term = st.selectbox("숙소 유형", ["장기", "단기"])
    amenities_input = st.text_area("편의시설 목록 (예: ['Wifi', 'TV', 'Kitchen'])")
    accommodates = st.number_input("수용 인원 수", min_value=1, value=2)
    availability_365 = st.number_input("연간 예약 가능 일수", min_value=0, max_value=365, value=180)

# ────────────────────────────────────────────────
# 4) 예측
if st.button("📊 슈퍼호스트 예측하기"):
    type_keywords = ['kitchen', 'tv', 'internet', 'wifi', 'air conditioning']
    common_keywords = ['essentials', 'heating', 'hot water', 'hangers', 'hair dryer']

    input_dict = {
        'host_response_time_score': convert_response_time(host_response_time),
        'host_acceptance_rate_score': convert_acceptance_rate(host_acceptance_rate),
        'host_has_profile_pic': convert_profile_pic(host_has_profile_pic),
        'host_identity_verified': convert_identity_verified(host_identity_verified),
        'is_long_term': convert_long_term(is_long_term),
        'type_amenities_score': convert_amenities(amenities_input, type_keywords),
        'common_amenities_score': convert_amenities(amenities_input, common_keywords),
        'accommodates': accommodates,
        'availability_365': availability_365
    }

    features = ['host_response_time_score', 'host_acceptance_rate_score',
                'host_has_profile_pic', 'host_identity_verified', 'is_long_term',
                'type_amenities_score', 'common_amenities_score',
                'accommodates', 'availability_365']

    X = pd.DataFrame([input_dict])[features]
    prediction = pipeline.predict(X)[0]

    result = "✅ 슈퍼호스트입니다!" if prediction == 1 else "❌ 슈퍼호스트가 아닙니다."
    st.subheader("예측 결과:")
    st.success(result)