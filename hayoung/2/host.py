import streamlit as st
import pandas as pd
import joblib

# --- 점수 변환 함수들 ---
def response_time_to_score(response_time_str):
    mapping = {
        'within an hour': 4,
        'within a few hours': 3,
        'within a day': 2,
        'a few days or more': 1
    }
    return mapping.get(response_time_str.lower(), 0)

def response_rate_to_score(rate_percent):
    rate = rate_percent / 100
    if rate <= 0.25:
        return 1
    elif rate <= 0.5:
        return 2
    elif rate <= 0.75:
        return 3
    else:
        return 4

def acceptance_rate_to_score(rate_percent):
    rate = rate_percent / 100
    if rate <= 0.25:
        return 1
    elif rate <= 0.5:
        return 2
    elif rate <= 0.75:
        return 3
    else:
        return 4

common_amenities = ['Carbon monoxide alarm', 'Essentials', 'Hangers', 'Smoke alarm', 'Wifi']
type_amenity_dict = {
    'high': ['Air conditioning', 'Building staff', 'Elevator', 'Gym', 'Heating', 'Paid parking off premises', 'Shampoo'],
    'low-mid': ['Cleaning products', 'Dining table', 'Exterior security cameras on property', 'Free street parking',
                'Freezer', 'Laundromat nearby', 'Lock on bedroom door', 'Microwave'],
    'mid': ['Cooking basics', 'Kitchen', 'Oven'],
    'upper-mid': ['Bathtub', 'Cleaning products', 'Cooking basics', 'Dishes and silverware', 'Elevator', 'Freezer']
}

def calc_amenity_scores(amenities_list, room_new_type):
    common_match = sum(1 for a in amenities_list if a in common_amenities) / len(common_amenities) if common_amenities else 0
    type_amenities = type_amenity_dict.get(room_new_type, [])
    type_match = sum(1 for a in amenities_list if a in type_amenities) / len(type_amenities) if type_amenities else 0
    return round(common_match, 3), round(type_match, 3)

# --- 모델 불러오기 ---
# 모델 파일 경로를 정확히 지정하세요.
try:
    # 예시 경로: C:/Users/HY/Documents/GitHub/advanced_project/hayoung/3/superhost_pipeline_rf.pkl
    pipeline = joblib.load('superhost_pipeline_rf.pkl') # 파일을 현재 스크립트와 같은 디렉토리에 두는 것을 권장합니다.
except FileNotFoundError:
    st.error("오류: 'superhost_pipeline_rf.pkl' 모델 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    st.stop() # 파일이 없으면 앱 실행 중지

st.title("슈퍼호스트 예측 앱")
st.write("아래 입력값을 채우고 '예측하기' 버튼을 누르세요.")

def main():
    # --- 입력 위젯들 ---
    st.header("호스트 정보")
    host_response_time = st.selectbox("호스트 응답 시간", ['within an hour', 'within a few hours', 'within a day', 'a few days or more'], help="게스트 문의에 응답하는 시간")
    host_response_rate = st.slider("호스트 응답률 (%)", 0, 100, 100, help="게스트 문의에 응답한 비율 (높을수록 좋음)")
    host_acceptance_rate = st.slider("호스트 수락률 (%)", 0, 100, 100, help="예약 요청을 수락한 비율 (높을수록 좋음)")
    host_about_length_group = st.selectbox("호스트 소개글 길이 그룹", ['short', 'medium', 'long'], index=2, help="프로필 소개글의 길이. 길수록 신뢰도 향상")
    host_has_profile_pic = st.radio("프로필 사진 있음?", [1, 0], format_func=lambda x: "있음" if x == 1 else "없음", index=0, help="프로필 사진 유무. 있을수록 슈퍼호스트 확률 높음")
    host_identity_verified = st.radio("호스트 신원 인증됨?", [1, 0], format_func=lambda x: "예" if x == 1 else "아니오", index=0, help="에어비앤비에서 신원 인증을 했는지 여부")

    st.header("숙소 정보")
    room_type = st.selectbox("방 타입", ['Entire home/apt', 'Private room', 'Shared room', 'Hotel room'], index=0, help="숙소의 주요 유형. 'Entire home/apt'가 유리")
    room_new_type = st.selectbox("숙소 가격대 그룹", ['high', 'low-mid', 'mid', 'upper-mid'], index=2, help="숙소의 대략적인 가격대. 'mid'가 선호됨")

    # 편의시설 선택 (st.multiselect 사용)
    # 모든 가능한 편의시설을 리스트로 합치기
    all_possible_amenities = sorted(list(set(common_amenities +
                                          type_amenity_dict['high'] +
                                          type_amenity_dict['low-mid'] +
                                          type_amenity_dict['mid'] +
                                          type_amenity_dict['upper-mid'] +
                                          ['TV', 'Dryer', 'Washer', 'Dishwasher', 'Coffee maker', 'Toaster', 'Iron', 'Hair dryer',
                                           'Bed linens', 'Extra pillows and blankets', 'First aid kit', 'Fire extinguisher', 'Locker'] # 기타 추가
                                         )))
    
    # 기본 선택될 편의시설 설정 (슈퍼호스트에 유리한 조건)
    default_amenities = [
        'Carbon monoxide alarm', 'Essentials', 'Hangers', 'Smoke alarm', 'Wifi', # 공통 필수
        'Cooking basics', 'Kitchen', 'Oven', # mid 타입 필수
        'Air conditioning', 'Heating', 'Shampoo', 'TV', 'Washer', 'Dryer', 'Hair dryer', 'Iron' # 기타 중요
    ]
    # 실제 존재하는 옵션들만 기본값으로 설정
    default_amenities = [a for a in default_amenities if a in all_possible_amenities]


    selected_amenities = st.multiselect(
        "제공하는 편의시설을 선택하세요 (다중 선택 가능)",
        options=all_possible_amenities,
        default=default_amenities, # 기본으로 선택될 값
        help="다양한 편의시설을 제공할수록 슈퍼호스트 확률이 높아집니다. 최소 37개 이상 권장."
    )
    # 선택된 편의시설 개수를 자동으로 반영 (사용자가 직접 입력하는 대신)
    amenities_cnt = len(selected_amenities)
    st.write(f"선택된 편의시설 개수: **{amenities_cnt}**") # 사용자에게 현재 선택 개수 표시

    availability_365 = st.number_input("연간 예약 가능일 수", min_value=0, max_value=365, value=233, help="1년 중 숙소를 예약 가능한 일수. 실제 운영 가능한 날짜만 열어두는 것이 좋습니다.")
    price = st.number_input("1박당 가격 ($)", min_value=0, value=129, help="숙소의 1박당 가격. 지역 평균 대비 ±10% 유지 권장")
    name_length_group = st.selectbox("숙소 이름 길이 그룹", ['short', 'medium', 'long'], index=2, help="숙소 이름의 길이. 길고 상세할수록 긍정적")
    description_length_group = st.selectbox("숙소 설명 길이 그룹", ['short', 'medium', 'long'], index=2, help="숙소 상세 설명의 길이. 길고 상세할수록 긍정적")
    is_long_term = st.radio("장기 숙박 가능 여부", [0, 1], format_func=lambda x: "아니오" if x == 0 else "예", index=0, help="장기 임대보다 단기 숙박 중심으로 운영하는 것이 유리")
    accommodates = st.number_input("최대 수용 인원", min_value=1, value=2, help="숙소에서 수용 가능한 최대 게스트 수")


    if st.button("슈퍼호스트 확률 예측하기"):
        # --- 점수 계산 ---
        response_time_score = response_time_to_score(host_response_time)
        response_rate_score = response_rate_to_score(host_response_rate)
        acceptance_rate_score = acceptance_rate_to_score(host_acceptance_rate)
        common_amenity_score, type_amenity_score = calc_amenity_scores(selected_amenities, room_new_type)

        # --- 예측을 위한 DataFrame 생성 ---
        # 이 DataFrame의 컬럼명과 순서는 pipeline 학습 시의 X_train 컬럼과 정확히 일치해야 합니다.
        # pipeline에 ColumnTransformer가 포함되어 One-Hot Encoding을 자동 처리한다면
        # 아래처럼 원본 형태의 데이터를 DataFrame으로 만들어도 됩니다.
        
        # 모델 학습 시의 컬럼 리스트 (pipeline 내부의 전처리기에서 자동으로 생성될 것을 가정)
        # 만약 pipeline이 ColumnTransformer + Estimator 조합이라면,
        # 입력 데이터는 원본 피처명으로 구성되어야 합니다.
        
        input_data_dict = {
            'amenities_cnt': amenities_cnt,
            'availability_365': availability_365,
            'price': price,
            'host_about_length_group': host_about_length_group, # 범주형
            'room_type': room_type,                               # 범주형
            'name_length_group': name_length_group,               # 범주형
            'description_length_group': description_length_group, # 범주형
            'host_has_profile_pic': host_has_profile_pic,
            'host_response_time_score': response_time_score,
            'type_amenity_score': type_amenity_score,
            'common_amenity_score': common_amenity_score,
            'host_acceptance_rate_score': acceptance_rate_score,
            'host_identity_verified': host_identity_verified,
            'is_long_term': is_long_term,
            'accommodates': accommodates
        }
        
        # DataFrame으로 변환
        new_data_df = pd.DataFrame([input_data_dict])

        # --- 예측 ---
        try:
            pred = pipeline.predict(new_data_df)
            proba = pipeline.predict_proba(new_data_df)[:, 1]

            st.subheader("예측 결과")
            if pred[0] == 1:
                st.success(f"**슈퍼호스트가 될 가능성이 높습니다!**")
                st.markdown(f"**슈퍼호스트 확률: <span style='color:green; font-size:1.5em;'>{round(proba[0]*100, 2)}%</span>**", unsafe_allow_html=True)
            else:
                st.warning(f"**슈퍼호스트가 아닐 가능성이 높습니다.**")
                st.markdown(f"**슈퍼호스트 확률: <span style='color:orange; font-size:1.5em;'>{round(proba[0]*100, 2)}%</span>**", unsafe_allow_html=True)
            
            st.info("슈퍼호스트 자격 유지를 위해 응답률/수락률 100%, 잦은 단기 예약, 다양한 편의시설 구비, 상세한 숙소 및 호스트 정보 제공 등이 중요합니다.")

        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {e}")
            st.write("모델 학습 시 사용된 변수들과 현재 입력된 변수들이 일치하는지 확인해 주세요.")
            st.write(f"입력된 데이터 컬럼: {new_data_df.columns.tolist()}")
            # st.write(f"모델의 예상 입력 컬럼 (preprocessor 사용 시 다를 수 있음): {pipeline.named_steps['preprocessor'].get_feature_names_out().tolist()}") # 파이프라인 구조에 따라 달라짐


if __name__ == "__main__":
    main()