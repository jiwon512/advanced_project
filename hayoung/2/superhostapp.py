# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1) 모델 로드
@st.cache_resource
def load_model(path):
    return joblib.load(path)

MODEL_PATH = 'superhost_pipeline_rf.pkl'  # 저장된 파이프라인 모델 경로
model = load_model(MODEL_PATH)

# 2) 사용자 입력 폼 만들기
st.title("슈퍼호스트 여부 예측 앱")

st.write("아래 정보를 입력하면 슈퍼호스트 여부를 예측합니다.")

# 입력 변수들
amenities_cnt = st.number_input("편의시설 개수 (amenities_cnt)", min_value=0, max_value=50, value=10)
availability_365 = st.number_input("연간 예약 가능 일수 (availability_365)", min_value=0, max_value=365, value=180)
price = st.number_input("가격 (price)", min_value=0.0, value=100.0)
host_about_length_group = st.selectbox("호스트 소개글 길이 (host_about_length_group)", ['short', 'medium', 'long'])
room_type = st.selectbox("숙소 유형 (room_type)", ['Entire home/apt', 'Private room', 'Shared room'])
name_length_group = st.selectbox("이름 길이 그룹 (name_length_group)", ['short', 'medium', 'long'])
description_length_group = st.selectbox("설명 길이 그룹 (description_length_group)", ['short', 'medium', 'long'])
host_has_profile_pic = st.selectbox("프로필 사진 있음? (host_has_profile_pic)", [0,1])
host_response_time_score = st.slider("호스트 응답 시간 점수 (1~4)", 1, 4, 3)
type_amenity_score = st.slider("타입별 편의시설 점수 (0.0~1.0)", 0.0, 1.0, 0.5, step=0.01)
common_amenity_score = st.slider("공통 편의시설 점수 (0.0~1.0)", 0.0, 1.0, 0.5, step=0.01)
host_acceptance_rate_score = st.slider("호스트 수락률 점수 (1~4)", 1, 4, 3)
host_identity_verified = st.selectbox("호스트 신원 인증 여부 (host_identity_verified)", [0,1])
is_long_term = st.selectbox("장기 예약 여부 (is_long_term)", [0,1])
accommodates = st.number_input("수용 인원 (accommodates)", min_value=1, max_value=20, value=2)

# 3) 입력값 DataFrame 만들기
input_dict = {
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

input_df = pd.DataFrame([input_dict])

# 4) 예측 버튼 및 결과 출력
if st.button("예측하기"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.write(f"**예측 결과: {'슈퍼호스트' if prediction==1 else '슈퍼호스트 아님'}**")
    st.write(f"**슈퍼호스트일 확률: {proba:.3f}**")
