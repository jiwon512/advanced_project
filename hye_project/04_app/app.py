import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re, ast

from mapping import (

    get_column_groups,
    row_new_to_old,
    model_features_old,
    cluster_map,
    borough_map,
    type_map,
    room_group_map as room_map,
    amen_group_map as amen_map,
    amen_selection_map
)

# ═════════════════════ 1) 데이터, 모델 로드 ══════════════════════
@st.cache_data
def load_df(path, usecols=None):
    return pd.read_csv(path, usecols=usecols)

@st.cache_resource
def load_pipeline(path):
    return joblib.load(path)

# ── hye data load  ─────────────────────────────────────────
hye_df_path     = "/hye_project/04_app/backup/processed_hye.csv"
hye_model_path  = "/Users/hyeom/Documents/GitHub/advanced_project/hye_project/04_app/price_prediction_pipeline.pkl"

hye_cols = get_column_groups()
model_features_new = hye_cols['model_features_new']
cat_cols       = hye_cols['cat_cols']
num_cols       = hye_cols['num_cols']
bin_cols       = hye_cols['bin_cols']
other_flags    = hye_cols['other_flags']

df = load_df(hye_df_path, usecols=get_column_groups()['ui_features'])
hye_pipeline  = load_pipeline(hye_model_path)
old_row = row_new_to_old(new_row_dict)
X = pd.DataFrame([old_row])[model_features_old]



# ── util : 예측 & Δ 계산 ─────────────────────────────────────────
def predict_price(row: dict) -> float:
    """row(dict) → USD 예측값 (열 순서 고정!)"""
    X = pd.DataFrame([row])[model_features]   # ★ 열 재정렬 핵심 ★
    return float(np.expm1(hye_pipeline.predict(X)[0]))

def add_strategy(bucket: list, label: str, test_row: dict, base: float):
    delta = predict_price(test_row) - base
    bucket.append((label, delta))

# ═════════════════════ 2) 기본 setting / defaults ════════════════════
defaults = {
    **df[num_cols].median().to_dict(),
    **df[cat_cols].mode().iloc[0].to_dict(),
    **{c:0 for c in bin_cols},
    **{f:0 for f in other_flags}
}

# 역매핑: 동네 -> 클러스터 코드
inv_cluster_map = {neigh: grp for grp, lst in cluster_map.items() for neigh in lst}

VAL_RMSE_USD = 48.36
MIN_NIGHTLY  = 10.0
MAX_NIGHTLY  = 900.0
max_acc      = int(df['accommodates'].max())

# ═════════════════════ 3) 공통 UI – 모드 선택 ════════════════════
st.title("🗽 NYC Airbnb 호스트 전략 도우미")
mode = st.radio("당신의 상태를 선택하세요", ["예비 호스트", "기존 호스트"])

# ═════════════════════ 4) 예비 호스트 ═══════════════════════════
if mode == "예비 호스트":
    key_prefix = "new_"          # ← 모든 위젯 key 에 prefix
    st.header("🚀 희망 수입 달성을 위한 맞춤 준비 가이드")

    # ── 4-1) 운영 지역(복수) ────────────────────────
    sel_boroughs = st.multiselect(
        "운영을 고려 중인 자치구(복수 선택 가능)",
        df['neighbourhood_group_cleansed'].unique().tolist(),
        default=["Manhattan"],
        key=f"{key_prefix}boroughs"
    )
    if not sel_boroughs:
        st.warning("최소 1개의 자치구를 선택해 주세요.")
        st.stop()

    # ── 4-2) 숙소 기본 사양 ─────────────────────────
    accommodates = st.slider("최대 숙박 인원", 1, max_acc, 2,
                             key=f"{key_prefix}accommodates")
    rt_ordinals  = sorted(df['room_type_ord'].unique())
    rt_labels    = [type_map[o] for o in rt_ordinals]
    rt_choice_lb = st.selectbox("희망 룸 타입", rt_labels,
                                key=f"{key_prefix}rt")
    rt_choice    = rt_ordinals[rt_labels.index(rt_choice_lb)]

    # ── 4-3) 목표 월 수익 → 목표 1박 요금 ───────────
    desired_month = st.number_input(
        "희망 월수입 ($)",
        0.0, 20000.0, 4000.0, 100.0,
        key=f"{key_prefix}desired_month"
    )
    open_days = st.number_input(
        "월 운영일 수", 1, 31, 30, key=f"{key_prefix}days")
    target_price = desired_month / open_days
    st.markdown(f"➡️ **목표 1박 요금** : `${target_price:,.0f}`")

    # ── 4-4) 추천 버튼 ──────────────────────────────
    if st.button("🔍 맞춤 추천 보기", key=f"{key_prefix}recommend"):
        with st.spinner("⏳ 추천 계산 중…"):
            base_row = {**defaults,
                        'accommodates'          : accommodates,
                        'room_type_ord'         : rt_choice,
                        'host_is_superhost'     : 0}

            recs = []
            for bor in sel_boroughs:
                cand_clusters = borough_map.get(bor, ['other'])[:5] or ['other']
                for cl in cand_clusters:
                    for amen_grp in df['amen_grp'].unique():
                        for new_ord in df['room_new_type_ord'].unique():
                            row = base_row | {
                                'neighbourhood_group_cleansed': bor,
                                'neigh_cluster_reduced'       : cl,
                                'amen_grp'                    : amen_grp,
                                'room_new_type_ord'           : new_ord
                            }
                            recs.append({
                                'Borough'  : bor,
                                'Cluster'  : cl,
                                'AmenGrp'  : amen_grp,
                                'NewGrp'   : new_ord,
                                'Pred'     : predict_price(row)
                            })
            rec_df = pd.DataFrame(recs)
            near   = rec_df.loc[rec_df['Pred'].sub(target_price).abs() <= 50]
            show   = near if not near.empty else rec_df.iloc[
                        rec_df['Pred'].sub(target_price).abs().sort_values().index[:10]]

            show['Cluster'] = show['Cluster'].map(lambda c: ','.join(cluster_map.get(c,[c])))
            show['NewGrp']  = show['NewGrp'].map(lambda o: ','.join(room_map.get(o,[str(o)])))
            show['Pred']    = show['Pred'].map(lambda x: f"${x:,.0f}")

            st.subheader("📋 추천 조합 (목표가 ±$50)")
            st.table(show.rename(columns={'Borough':'자치구','Cluster':'지역 클러스터',
                                          'AmenGrp':'Amenity 그룹','NewGrp':'신규 룸그룹',
                                          'Pred':'예측 1박 요금'}))

            # ── 추가 팁 ────────────────────────────────
            st.subheader("💡 준비 팁")
            samp_row   = rec_df.iloc[0]
            base_price = samp_row['Pred']
            sh_row     = base_row | {'host_is_superhost':1,
                                     'neighbourhood_group_cleansed': samp_row['Borough'],
                                     'neigh_cluster_reduced'       : samp_row['Cluster'],
                                     'amen_grp'                    : samp_row['AmenGrp'],
                                     'room_new_type_ord'           : samp_row['NewGrp']}
            delta_sh = predict_price(sh_row) - base_price
            st.markdown(f"- **슈퍼호스트 달성 시** 예상 ↑ **${delta_sh:,.0f}** /박")

            real_match = df[(df['price'].between(target_price-50, target_price+50)) &
                            (df['neighbourhood_group_cleansed'].isin(sel_boroughs))]
            if not real_match.empty:
                avg_cnt = int(real_match['amenities_cnt'].mean())
                st.markdown(f"- 해당 가격대 평균 Amenity 수 **{avg_cnt}개** 이상 준비 권장")
            else:
                st.markdown("- (해당 가격대 실거래 데이터 부족)")

# ═════════════════════ 5) 기존 호스트 ═══════════════════════════
if mode == "기존 호스트":
    key_prefix = "old_"
    st.header("📈 기존 호스트를 위한 전략 분석")
    profile = {}

    # ── 5-1) 프로필 입력 ─────────────────────────────
    # 자치구 & 동네
    bor = st.selectbox("자치구(Borough)",
                       df['neighbourhood_group_cleansed'].unique(),
                       key=f"{key_prefix}bor")
    profile['neighbourhood_group_cleansed'] = bor

    neigh_list   = borough_map.get(bor, [])
    neigh_counts = df[df['neighbourhood_group_cleansed']==bor]\
                     ['neighbourhood_cleansed'].value_counts()
    neigh_sorted = sorted(neigh_list, key=lambda x: neigh_counts.get(x,0), reverse=True)
    neigh = st.selectbox("동네(Neighborhood)", neigh_sorted, key=f"{key_prefix}neigh")
    profile['neigh_cluster_reduced'] = inv_cluster_map.get(neigh, 'other')

    # 룸 타입, 구조
    rt_ordinals = sorted(df['room_type_ord'].unique())
    rt_labels   = [type_map[o] for o in rt_ordinals]
    rt_lb       = st.selectbox("룸 타입", rt_labels, key=f"{key_prefix}rt")
    profile['room_type_ord'] = rt_ordinals[rt_labels.index(rt_lb)]

    struct_opts = df[df['room_type_ord']==profile['room_type_ord']]['room_structure_type']\
                    .value_counts().index.tolist()
    struct = st.selectbox("숙소 구조(설명란)", struct_opts, key=f"{key_prefix}struct")
    profile['room_structure_type'] = struct
    inv_room = {s:g for g, lst in room_map.items() for s in lst}
    profile['room_new_type_ord'] = inv_room.get(struct,0)

    with st.expander("숙소 설명란은 어떻게 선택하나요? 📋", expanded=False):
        st.image(
            "/Users/hyeom/Documents/GitHub/advanced_project/hye_project/structure_example.png",
            use_container_width=True
        )
        st.write("숙소 설명란의 해당 정보를 바탕으로 작성해 주세요! 임의로 선택시 예측율이 떨어질 수 있어요.")

    # 숙박 인원
    acc = st.number_input("최대 숙박 인원", 1, max_acc,
                          int(defaults['accommodates']), 1,
                          key=f"{key_prefix}acc")
    profile['accommodates'] = acc

    # 예약/정책 토글
    profile['instant_bookable']   = int(st.toggle("Instant Bookable",
                                       key=f"{key_prefix}inst"))
    profile['is_long_term']       = int(st.toggle("장기 숙박 허용",
                                       key=f"{key_prefix}long"))
    profile['host_is_superhost']  = int(st.toggle("슈퍼호스트 여부",
                                       key=f"{key_prefix}super"))

    # Amenity 멀티선택
    def clean(s:str)->str: return re.sub(r'[\uD800-\uDFFF]', '', s).lower().strip()
    grp_label = {0:'low-mid',1:'mid',2:'upper-mid',3:'high'}.get(
                    profile['room_new_type_ord'],'common')
    default_opts = [a for a in amen_selection_map
                    if clean(a) in [clean(x) for x in amen_map[grp_label]]]
    sel_am = st.multiselect("주요 Amenity", amen_selection_map, default_opts,
                            key=f"{key_prefix}amen")
    profile['amenities_cnt'] = len(sel_am)
    profile['amen_grp']      = grp_label
    for flag in ['air conditioning','wifi','bathtub',
                 'carbon monoxide alarm','elevator']:
        profile[f"has_{flag.replace(' ','_')}"] = int(flag in map(clean, sel_am))

    # ───────────────────────────────────────────────
    # 5-2) 성과/목표 입력
    st.subheader("📊 성과 및 목표 입력")
    booked_days = st.number_input("한 달 예약된 날 수", 1, 31, 20,
                                  key=f"{key_prefix}days")
    MIN_REV = booked_days*MIN_NIGHTLY
    MAX_REV = booked_days*MAX_NIGHTLY

    curr_rev = st.number_input("현재 월 수익 ($)", MIN_REV, MAX_REV, 3000.0, 50.0,
                               key=f"{key_prefix}curr")
    desired_rev = st.number_input("목표 월 수익 ($)", MIN_REV, MAX_REV, 4000.0, 50.0,
                                  key=f"{key_prefix}goal")

    curr_adr = curr_rev/booked_days
    target_adr = desired_rev/booked_days
    st.metric("현재 ADR", f"${curr_adr:,.0f}")
    st.metric("목표 ADR", f"${target_adr:,.0f}", f"${target_adr-curr_adr:,.0f}")

    with st.expander("💡 팁: ADR(1박 평균요금)이란?"):
        st.write("ADR = (한 달 총수익) ÷ (한 달 예약된 날 수)로, 수익 목표를 달성하려면, 이 ADR 값을 방 가격 설정의 기준으로 활용하세요.")

    # ───────────────────────────────────────────────
    # 5-3) 비교 모드 & 버튼
    compare_mode = st.selectbox(
        "💡 목표 비교 방식",
        ["min→max (구간 최소→목표 최대)",
         "max→max (구간 최대→목표 최대)",
         "mean→mean (평균↔평균)"],
        index=1, key=f"{key_prefix}mode"
    )

    if st.button("🔍 분석 시작", key=f"{key_prefix}run"):
        if not (10<=curr_adr<=900 and 10<=target_adr<=900):
            st.error("평균 1박 가격은 $10 ~ $900 사이여야 합니다.")
            st.stop()

        with st.spinner("⏳ 분석 중…"):
            base_row       = {**defaults, **profile}
            row_cur        = base_row | {'curr_avg_price': curr_adr}
            row_tar        = base_row | {'curr_avg_price': target_adr}
            pred_cur       = predict_price(row_cur)
            pred_tar       = predict_price(row_tar)

            cur_lo, cur_hi = pred_cur-VAL_RMSE_USD, pred_cur+VAL_RMSE_USD
            tar_lo, tar_hi = pred_tar-VAL_RMSE_USD, pred_tar+VAL_RMSE_USD
            cur_mu, tar_mu = (cur_lo+cur_hi)/2, (tar_lo+tar_hi)/2

            if compare_mode.startswith("min"):
                need = max(0, tar_hi - cur_lo)
            elif compare_mode.startswith("max"):
                need = max(0, tar_hi - cur_hi)
            else:
                need = max(0, tar_mu - cur_mu)

            st.markdown(
                f"**현재 구간** : ${cur_lo:,.0f} ~ ${cur_hi:,.0f} (평균 ${cur_mu:,.0f})  \n"
                f"**목표 구간** : ${tar_lo:,.0f} ~ ${tar_hi:,.0f} (평균 ${tar_mu:,.0f})"
            )
            if need==0:
                st.success("이미 목표 구간에 도달했습니다! 🎉")
                st.stop()
            else:
                st.write(f"→ **${need:,.0f}** ↑ 필요 ({compare_mode.split()[0]} 기준)")

            # ── 전략 집계 ───────────────────────────
            buckets = {"🛠 Host Quality":[], "🌿 Guest Experience":[], "🔧 Amenities Upgrade":[]}
            base_pred = pred_cur

            # Host Quality
            if not profile['host_is_superhost']:
                add_strategy(buckets["🛠 Host Quality"], "슈퍼호스트 달성",
                             row_cur | {'host_is_superhost':1}, base_pred)
            add_strategy(buckets["🛠 Host Quality"], "리뷰 평점 +0.5",
                         row_cur | {'review_scores_rating': min(defaults['review_scores_rating']+0.5,5)},
                         base_pred)

            # Guest Experience
            add_strategy(buckets["🌿 Guest Experience"], "장기체류 허용",
                         row_cur | {'is_long_term':1}, base_pred)
            for inc in (1,2):
                add_strategy(buckets["🌿 Guest Experience"], f"숙박 인원 +{inc}",
                             row_cur | {'accommodates': acc+inc}, base_pred)

            # Amenities
            for alt in df['amen_grp'].unique():
                if alt!=profile['amen_grp']:
                    add_strategy(buckets["🔧 Amenities Upgrade"],
                                 f"Amenity 그룹 → {alt}",
                                 row_cur | {'amen_grp':alt}, base_pred)
            for inc in (3,5):
                add_strategy(buckets["🔧 Amenities Upgrade"],
                             f"Amenity 개수 +{inc}",
                             row_cur | {'amenities_cnt': profile['amenities_cnt']+inc},
                             base_pred)

            # pick until need 충족
            flat = [(sec,lbl,d) for sec,v in buckets.items() for lbl,d in v if d>0]
            flat.sort(key=lambda x:x[2], reverse=True)
            picks, cum = [],0
            for sec,lbl,d in flat:
                picks.append((sec,lbl,d)); cum+=d
                if cum>=need: break

            st.subheader("🔧 갭 해소 추천 전략")
            if not picks:
                st.info("적절한 전략을 찾지 못했습니다 🤔")
            else:
                cur_sec=None
                for sec,lbl,d in picks:
                    if sec!=cur_sec:
                        st.markdown(f"**{sec}**")
                        cur_sec=sec
                    st.markdown(f"- {lbl} **(+${d:,.0f})**")
                st.success(f"예상 상승 +${cum:,.0f} ≥ 필요 +${need:,.0f}")

# ───────────────────────────────────────────────────────────────
