import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─── 1) 데이터 & 모델 로드 ───────────────────────────────────────────────
@st.cache_data
def load_df(path='df.csv'):
    return pd.read_csv('/Users/hyeom/Documents/GitHub/advanced_project/hye_project/for_machine_learning_2.csv')

df = load_df()
model = joblib.load('/Users/hyeom/Documents/GitHub/advanced_project/hye_project/03_MachineLearning/final_ensemble_model_2.pkl')

# ─── 2) 피처 정의 ────────────────────────────────────────────────────────
cat_cols = [
    'neigh_cluster_reduced','neighbourhood_group_cleansed',
    'room_type_ord','room_new_type_ord','room_structure_type','amen_grp',
    'description_length_group','name_length_group'
]
num_cols = [
    'latitude','longitude','accommodates','bath_score_mul',
    'amenities_cnt','review_scores_rating',
    'number_of_reviews','number_of_reviews_ltm','region_score_norm',
    'host_response_time_score','host_response_rate_score'
]
bin_cols = [
    'instant_bookable','is_long_term','host_is_superhost',
    'has_Air_conditioning','has_Wifi',
    'has_Bathtub','has_Carbon_monoxide_alarm','has_Elevator',
    'neighborhood_overview_exists'
]
other_flags = ['grp01_high','grp04_high']
features = cat_cols + num_cols + bin_cols + other_flags

# ─── 모델 분포 & RMSE 기반 범위 계산 ──────────────────────────────────────
pred_log = model.predict(df[features])  # 기존 모델이 학습된 피처 리스트 사용
pred_price = np.expm1(pred_log)
q1, q3 = np.percentile(pred_price, [25, 75])
rmse_usd = 48.92  # 모델 RMSE($)
lower_bound = max(pred_price.min(), q1 - rmse_usd)
upper_bound = q3 + rmse_usd
mean_price = pred_price.mean()

# 기본값 세팅 (중립값)
median_vals = df[num_cols].median()
mode_cat = df[cat_cols].mode().iloc[0]
defaults = {
    **mode_cat.to_dict(),
    **median_vals.to_dict(),
    **{c:0 for c in bin_cols},
    **{f:0 for f in other_flags}
}

# ─── 3) UI ────────────────────────────────────────────────────────────────
st.title("🗽 NYC Airbnb 호스트 전략 도우미")
mode = st.radio("당신의 상태를 선택하세요", ["예비 호스트","기존 호스트"])

if mode == "예비 호스트":
    st.header("🚀 희망 수입에 따라 준비해보세요!")

    # 목표 기준 선택
    choice = st.radio("준비 방법 추천 기준을 선택하세요", ["1박 요금","월수입"])
    if choice == "1박 요금":
        target_price = st.number_input(
            "원하는 1박 요금 ($)",
            min_value=10.0,               # ← 최소 10 달러
            max_value=900.0,              # ← 최대 900 달러
            value=min(max(mean_price, 10.0), 900.0),  # 기본값도 범위 내로 설정
            step=1.0,
            help="희망 1박 요금은 10달러 ~ 900달러 사이에서 입력해 주세요."
        )
    else:
        desired_monthly = st.number_input(
            "원하는 월수입 ($)",
            min_value=0.0,
            value=3000.0,
            step=100.0
        )
        occ_rate = st.slider(
            "예상 예약율",
            0.0,1.0,0.7,0.01,
            help="예: 70% 예약율은 0.7로 입력"
        )
        days = st.number_input(
            "운영 기간(일)",
            min_value=1,
            max_value=365,
            value=30
        )
        target_price = desired_monthly / (occ_rate * days)

    st.write(f"▶️ 목표 1박 요금: **${target_price:,.0f}**")

    # 추천 함수
    def recommend(feature, candidates, top_n=5):
        preds = {}
        for val in candidates:
            row = defaults.copy()
            row.update({feature: val, 'host_is_superhost': 0})
            df_row = pd.DataFrame([row])[features]
            p = np.expm1(model.predict(df_row)[0])
            preds[val] = p
        df_rec = (
            pd.DataFrame.from_dict(preds, orient='index', columns=['pred_price'])
              .query("pred_price >= @target_price")
              .sort_values('pred_price', ascending=False)
              .head(top_n)
        )
        return df_rec

    # 각 피처별 추천 TOP5
    st.subheader("🏘️ 추천 지역 그룹")
    st.table(recommend('neighbourhood_group_cleansed', df['neighbourhood_group_cleansed'].unique()))

    st.subheader("📍 추천 지역 클러스터")
    st.table(recommend('neigh_cluster_reduced', df['neigh_cluster_reduced'].unique()))

    st.subheader("🛎️ 추천 Amenity 그룹")
    st.table(recommend('amen_grp', df['amen_grp'].unique()))

    st.subheader("🛏️ 추천 룸 타입")
    st.table(recommend('room_type_ord', df['room_type_ord'].unique()))

    st.subheader("🏠 추천 신규 룸 타입")
    st.table(recommend('room_new_type_ord', df['room_new_type_ord'].unique()))

    st.subheader("📋 준비해야 할 사항")
    st.write("""
    - 고화질 사진 & 상세 설명  
    - 동적 가격(Price Surge) 전략 준비  
    - 빠른 응답으로 예약율↑  
    - 슈퍼호스트 자격 요건 확인  
    - 편의시설 & 리뷰 평점 강화
    """)

else:
    st.header("📈 기존 호스트를 위한 수익 개선 전략")

    # ─── 현황 입력 (입력값 검증 포함) ────────────────────────────────────────
    # 1) 가격 입력 (10~1000 범위로 제한, 도움말 추가)
    curr_price = st.number_input(
        "현재 1박당 요금 ($)",
        min_value=10.0,
        max_value=1000.0,
        value=50.0,
        step=1.0,
        help="호스트 요금은 10달러 이상, 1000달러 이하로 입력하세요."
    )

    # 2) 예약율 입력 (0~1 범위, 도움말 추가)
    curr_occ = st.slider(
        "현재 예약율",
        0.0,
        1.0,
        0.7,
        0.01,
        help="예: 70% 예약율은 0.7로 입력"
    )

    # 3) 운영 기간 입력 (1~365일 범위)
    days = st.number_input(
        "운영 기간(일)",
        min_value=1,
        max_value=365,
        value=30,
        help="1일부터 365일까지 입력 가능"
    )

    # 4) 목표 추가 수입 (0~10000 범위)
    add_inc = st.number_input(
        "추가 목표 수입 ($)",
        min_value=0.0,
        max_value=10000.0,
        value=500.0,
        step=50.0,
        help="최대 10,000달러까지 설정 가능"
    )

    # 5) 입력값 검증
    errors = []
    if curr_price < 10 or curr_price > 1000:
        errors.append("요금은 10~1000 사이여야 합니다.")
    if days < 1 or days > 365:
        errors.append("운영 기간은 1~365일 사이여야 합니다.")
    if add_inc < 0 or add_inc > 10000:
        errors.append("추가 목표 수입은 0~10000 사이여야 합니다.")

    if errors:
        for e in errors:
            st.error(e)
        st.stop()  # 오류가 있으면 이하 로직 실행 중단

    if st.button("전략 추천"):
        strategies = []

        # 1) 요금 +5%
        p5   = curr_price * 1.05
        rev5 = p5 * curr_occ * days
        strategies.append({"action":"요금 +5%","new_price":f"${p5:,.0f}","income":f"${rev5:,.0f}"})

        # 2) 예약율 +5%
        o5    = min(1.0, curr_occ + 0.05)
        rev_o5= curr_price * o5 * days
        strategies.append({"action":"예약율 +5%","new_price":f"${curr_price:,.0f}","income":f"${rev_o5:,.0f}"})

        # 3) 슈퍼호스트 전환
        row = defaults.copy(); row['host_is_superhost'] = 1
        price_sh = np.expm1(model.predict(pd.DataFrame([row])[features])[0])
        rev_sh   = price_sh * curr_occ * days
        strategies.append({"action":"슈퍼호스트 달성","new_price":f"${price_sh:,.0f}","income":f"${rev_sh:,.0f}"})

        # 4) 리뷰 평점 +0.5
        row = defaults.copy();
        row['review_scores_rating'] = min(5.0, defaults['review_scores_rating'] + 0.5)
        price_rp = np.expm1(model.predict(pd.DataFrame([row])[features])[0])
        rev_rp   = price_rp * curr_occ * days
        strategies.append({"action":"리뷰 평점 +0.5","new_price":f"${price_rp:,.0f}","income":f"${rev_rp:,.0f}"})

        st.subheader("추천 전략별 예상 수입")
        st.table(pd.DataFrame(strategies))

# ─────────────────────────────────────────────────────────────────────────────
# 한 줄 요약:
# “목표 요금 기준으로 주요 피처별 상위 후보를 예측·추천하고, 기존 호스트는 요금·예약율·슈퍼호스트·리뷰 개선 전략별 예상 수입을 보여줍니다.”
