import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

# ───────────────────────────────────────────────────────
# 1. 모델 및 학습에 사용한 feature 순서 불러오기
# ───────────────────────────────────────────────────────
@st.cache_resource
def load_model(path):
    return joblib.load(path)

pipeline = load_model("for_app.pkl")

# 학습 당시 사용한 컬럼 순서
features = joblib.load("train_columns.pkl")

# ───────────────────────────────────────────────────────
# 2. 점수 변환 함수
# ───────────────────────────────────────────────────────
def response_time_to_score(x):
    mapping = {
        "within an hour": 4,
        "within a few hours": 3,
        "within a day": 2,
        "a few days or more": 1
    }
    return mapping.get(x.lower(), 0)

def acceptance_rate_to_score(rate):
    try:
        rate = int(rate.strip('%'))
    except:
        return 0
    if rate >= 95:
        return 4
    elif rate >= 90:
        return 3
    elif rate >= 80:
        return 2
    else:
        return 1

# ───────────────────────────────────────────────────────
# 3. 편의시설 점수 계산
# ───────────────────────────────────────────────────────
common_amenities = ['Wifi', 'Kitchen', 'Heating', 'Washer', 'Hangers', 'Hair dryer', 'Iron']
type_amenities = {
    'Entire home/apt': ['TV', 'Coffee maker', 'Microwave', 'Refrigerator'],
    'Private room': ['Desk', 'Shampoo', 'Body soap'],
    'Shared room': ['Lock on bedroom door', 'First aid kit']
}

def calculate_amenity_scores(selected_amenities, room_type):
    common_score = np.mean([a in selected_amenities for a in common_amenities])
    type_list = type_amenities.get(room_type, [])
    type_score = np.mean([a in selected_amenities for a in type_list])
    return round(common_score, 2), round(type_score, 2)

# ───────────────────────────────────────────────────────
# 4. Streamlit UI
# ───────────────────────────────────────────────────────
st.title("Airbnb 슈퍼호스트 예측기")
st.write("입력값을 기반으로 슈퍼호스트 여부를 예측합니다.")

col1, col2 = st.columns(2)

with col1:
    host_response_time = st.selectbox("호스트 응답 시간", ["within an hour", "within a few hours", "within a day", "a few days or more"])
    host_acceptance_rate = st.text_input("호스트 수락률 (예: 96%)", "96%")
    host_has_profile_pic = st.radio("프로필 사진 여부", ["있음", "없음"])
    host_identity_verified = st.radio("신분 인증 여부", ["있음", "없음"])
    is_long_term = st.radio("임대 유형", ["장기", "단기"])

with col2:
    room_type = st.selectbox("숙소 유형", ["Entire home/apt", "Private room", "Shared room"])
    price = st.number_input("가격 (1박 기준)", min_value=10, max_value=1000, value=100)
    amenities_input = st.multiselect(
        "편의시설 선택",
        options=sorted(set(common_amenities + sum(type_amenities.values(), [])))
    )

# 변환
response_score = response_time_to_score(host_response_time)
acceptance_score = acceptance_rate_to_score(host_acceptance_rate)
profile_pic_flag = 1 if host_has_profile_pic == "있음" else 0
id_verified_flag = 1 if host_identity_verified == "있음" else 0
long_term_flag = 1 if is_long_term == "장기" else 0
common_score, type_score = calculate_amenity_scores(amenities_input, room_type)

# 입력 dict 구성
input_dict = {
    'host_response_time_score': response_score,
    'host_acceptance_rate_score': acceptance_score,
    'host_has_profile_pic': profile_pic_flag,
    'host_identity_verified': id_verified_flag,
    'is_long_term': long_term_flag,
    'room_type': room_type,
    'price': price,
    'common_amenity_score': common_score,
    'type_amenity_score': type_score,
}

# 누락된 나머지 feature를 기본값으로 채움
full_input = {col: 0 for col in features}  # 모든 값 0 초기화
full_input.update(input_dict)  # 입력값 반영
X = pd.DataFrame([full_input])[features]

# 예측
if st.button("예측하기"):
    pred = pipeline.predict(X)[0]
    label = "✅ 슈퍼호스트입니다!" if pred == 1 else "❌ 슈퍼호스트가 아닙니다."
    st.subheader(label)
