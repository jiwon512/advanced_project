import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 전처리 및 통계 함수 import
from hostapp2 import (
    group_host_about_length, group_name_length, group_description_length,
    response_time_to_score, response_rate_to_score, acceptance_rate_to_score,
    calc_amenity_scores
)

# 데이터 및 모델 로드
@st.cache_data
def load_data():
    df = pd.read_csv('superhost.csv')
    return df

@st.cache_resource
def load_model():
    with open('superhost_pipeline_rf.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

df = load_data()
model = load_model()

# --- Streamlit UI ---
st.title("Airbnb Superhost 예측 및 통계 대시보드")

# 1. 유저 입력 폼
with st.form("input_form"):
    host_about = st.text_area("호스트 소개글")
    name = st.text_input("숙소 이름")
    description = st.text_area("숙소 설명")
    response_time = st.selectbox("응답 시간", ["within an hour", "within a few hours", "within a day", "a few days or more"])
    response_rate = st.slider("응답률(%)", 0, 100, 90)
    acceptance_rate = st.slider("수락률(%)", 0, 100, 95)
    amenities = st.multiselect("어메니티", options=df['amenities'].explode().unique())
    room_type = st.selectbox("방 타입", df['room_new_type'].unique())
    submitted = st.form_submit_button("예측 및 통계 보기")

# 2. 통계값 시각화
st.header("전체 데이터 통계")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("평균 응답률", f"{df['host_response_rate'].mean():.2f}%")
    st.metric("평균 수락률", f"{df['host_acceptance_rate'].mean():.2f}%")
with col2:
    st.metric("중앙값 응답률", f"{df['host_response_rate'].median():.2f}%")
    st.metric("중앙값 수락률", f"{df['host_acceptance_rate'].median():.2f}%")
with col3:
    st.metric("Superhost 비율", f"{df['superhost'].mean()*100:.2f}%")

# 3. 예측 및 결과 해석
if submitted:
    # 입력값 전처리
    about_len = len(host_about)
    name_len = len(name)
    desc_len = len(description)
    about_group = group_host_about_length(about_len)
    name_group = group_name_length(name_len)
    desc_group = group_description_length(desc_len)
    resp_time_score = response_time_to_score(response_time)
    resp_rate_score = response_rate_to_score(response_rate)
    acc_rate_score = acceptance_rate_to_score(acceptance_rate)
    amenity_score = calc_amenity_scores(amenities, room_type)
    
    # 모델 입력 포맷에 맞게 데이터 구성
    input_df = pd.DataFrame([{
        'about_group': about_group,
        'name_group': name_group,
        'desc_group': desc_group,
        'resp_time_score': resp_time_score,
        'resp_rate_score': resp_rate_score,
        'acc_rate_score': acc_rate_score,
        'amenity_score': amenity_score,
        # ... 필요한 추가 feature
    }])
    
    # 예측
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    
    st.subheader("예측 결과")
    st.write(f"Superhost 예측: {'Yes' if pred else 'No'} (확률: {proba:.2%})")
    
    # 입력값 분위/백분위 시각화
    st.subheader("입력값의 데이터 내 위치")
    st.write("응답률 분위:", np.searchsorted(np.sort(df['host_response_rate']), response_rate) / len(df))
    st.write("수락률 분위:", np.searchsorted(np.sort(df['host_acceptance_rate']), acceptance_rate) / len(df))
    # ... 추가 시각화

st.caption("데이터 및 모델: Airbnb NYC, RandomForest 기반 Superhost 예측")