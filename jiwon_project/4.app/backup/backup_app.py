# app.py  ─────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re, ast

# occupancy 예측용 피처리스트
occ_cols = [
 'host_is_superhost', 'host_total_listings_count', 'host_has_profile_pic', 'accommodates', 'beds',
 'availability_365', 'is_long_term', 'amenities_cnt', 'neighborhood_overview_exists',
 'name_length_group', 'description_length_group', 'host_about_length_group', 'host_location_ny',
 'is_private', 'bath_score_mul', 'is_activate', 'log_price', 'room_new_type_encoded',
 'neighbourhood_cluster', 'poi_pca', 'host_response_pca', 'host_verifications_count', 'score_info_pca'
]

# occupancy 예측 모델/데이터 경로
OCC_MODEL_PATH = "/Users/Jiwon/Documents/GitHub/advanced_project/jiwon_project/4.app/backup/backup_app.py"
OCC_DF_PATH = "/Users/Jiwon/Documents/GitHub/advanced_project/jiwon_project/presentation/jiwon_entire.csv"

# occupancy 예측 데이터/모델 로드
@st.cache_data
def load_occ_df(path):
    return pd.read_csv(path)
@st.cache_resource
def load_occ_model(path):
    return joblib.load(path)
occ_df = load_occ_df(OCC_DF_PATH)
occ_model = load_occ_model(OCC_MODEL_PATH)

# occupancy 예측 함수
def predict_occupancy(row: dict) -> float:
    X = pd.DataFrame([row])[occ_cols].fillna(0)
    return float(occ_model.predict(X)[0])

# ═════════════════════ 1) 데이터, 모델 로드 ══════════════════════
@st.cache_data
def load_df(path):
    return pd.read_csv(path)

@st.cache_resource
def load_pipeline(path):
    return joblib.load(path)

DF_PATH      = "/Users/hyeom/Documents/GitHub/advanced_project/hye_project/for_machine_learning_2.csv"
MODEL_PATH   = "/Users/hyeom/Documents/GitHub/advanced_project/hye_project/03_MachineLearning/for_app.pkl"

df           = load_df(DF_PATH)
pipeline     = load_pipeline(MODEL_PATH)


# ── util : 예측 & Δ 계산 ─────────────────────────────────────────
def predict_price(row: dict) -> float:
    """row(dict) → USD 예측값 (열 순서 고정!)"""
    X = pd.DataFrame([row])[hye_features]   # ★ 열 재정렬 핵심 ★
    return float(np.expm1(pipeline.predict(X)[0]))


def add_strategy(bucket: list, label: str, test_row: dict, base: float):
    delta = predict_price(test_row) - base
    bucket.append((label, delta))


# ═════════════════════ 2) 기본 setting / 매핑 ════════════════════
cat_cols = ['neigh_cluster_reduced','neighbourhood_group_cleansed',
            'room_type_ord','room_new_type_ord','room_structure_type',
            'amen_grp','description_length_group','name_length_group']
num_cols = ['latitude','longitude','accommodates','bath_score_mul','amenities_cnt',
            'review_scores_rating','number_of_reviews','number_of_reviews_ltm',
            'region_score_norm','host_response_time_score','host_response_rate_score']
bin_cols = ['instant_bookable','is_long_term','host_is_superhost',
            'has_Air_conditioning','has_Wifi','has_Bathtub',
            'has_Carbon_monoxide_alarm','has_Elevator',
            'neighborhood_overview_exists']
other_flags = ['grp01_high','grp04_high']

# 0) 학습 때 쓰던 정확한 순서
hye_features = cat_cols + num_cols + bin_cols + other_flags   # ← 전역에 추가

defaults = {
    **df[num_cols].median().to_dict(),
    **df[cat_cols].mode().iloc[0].to_dict(),
    **{c:0 for c in bin_cols},
    **{f:0 for f in other_flags}
}

# 매핑: 신규 룸 타입 -> 구조 리스트
room_map = {
    1: ['rental unit', 'guest suite', 'place', 'townhouse', 'serviced apartment', 'guesthouse'], #mid
    2: ['condo', 'loft', 'houseboat', 'boutique hotel', 'boat', 'villa', 'tiny home', 'bungalow', #upper-mid
                  'cottage', 'aparthotel', 'barn'],
    0: ['home', 'bed and breakfast', 'casa particular', 'vacation home', 'earthen home', 'camper/rv', #low-mid
                'hostel', 'kezhan', 'ranch', 'religious building', 'dome'],
    3: ['hotel', 'resort', 'tower'] # high
}
# 매핑: 클러스터 코드 -> 동네 목록
cluster_map = {
    'nbr_grp_04': ['Prospect Heights', 'Williamsburg', "Hell's Kitchen", 'Fort Greene', 'Clinton Hill',
                   'Chelsea', 'Gowanus', 'Lower East Side', 'East Village', 'Park Slope', 'Upper East Side',
                   'Middle Village', 'South Slope', 'Upper West Side', 'Chinatown', 'Windsor Terrace',
                   'Prospect-Lefferts Gardens', 'Downtown Brooklyn', 'Long Island City', 'Spuyten Duyvil',
                   'Gramercy', 'Lighthouse Hill', 'Springfield Gardens', 'Little Italy', 'New Brighton',
                   'Howland Hook', 'Roosevelt Island', 'Pelham Bay', 'East Morrisania', 'Mill Basin',
                   'Bergen Beach', "Prince's Bay", 'Navy Yard', 'Gerritsen Beach', 'Breezy Point',
                   'University Heights', 'West Farms', 'Oakwood', 'Dongan Hills', 'Grymes Hill'],
    'nbr_grp_03': ['East Harlem', 'Bedford-Stuyvesant', 'Crown Heights', 'Mott Haven', 'Morningside Heights',
                   'Rockaway Beach', 'Eastchester', 'Sheepshead Bay', 'East New York', 'Two Bridges',
                   'City Island', 'Port Morris', 'Arverne', 'Queens Village', 'Canarsie', 'Bay Terrace',
                   'Forest Hills', 'Unionport', 'Jamaica', 'Bayside', 'South Ozone Park', 'Howard Beach',
                   'Fresh Meadows', 'Bellerose', 'Edgemere', 'Stuyvesant Town', 'Rosedale', 'Kew Gardens Hills',
                   'Laurelton', 'Tremont', 'Olinville', 'College Point', 'Westchester Square',
                   'North Riverdale', 'Douglaston', 'Far Rockaway', 'Cambria Heights', 'Jamaica Hills',
                   'Woodlawn', 'Castle Hill', 'Van Nest', 'Country Club', 'Riverdale'],
    'nbr_grp_05': ['Harlem', 'Washington Heights', 'Ditmars Steinway', 'Astoria', 'Ridgewood', 'Clason Point',
                   'Kingsbridge', 'Bushwick', 'Sunnyside', 'Kensington', 'Briarwood', 'Allerton', 'Flushing',
                   'East Elmhurst', 'Norwood', 'Concourse', 'Richmond Hill', 'Maspeth', 'Soundview',
                   'Rego Park', 'Woodhaven', 'Mount Hope', 'Concourse Village', 'Midwood', 'Ozone Park',
                   'Cypress Hills', 'Manhattan Beach', 'Brownsville', 'Holliswood', 'Baychester', 'Wakefield',
                   'St. Albans', 'Whitestone', 'Mount Eden', 'Glendale', 'Morrisania', 'Marble Hill', 'Hollis',
                   'Williamsbridge', 'Melrose', 'Throgs Neck', 'Parkchester', 'Schuylerville', 'Belmont',
                   'Morris Heights', 'Little Neck'],
    'nbr_grp_01': ['Carroll Gardens', 'Midtown', 'Greenpoint', 'West Village', 'Brooklyn Heights', 'Kips Bay',
                   'Nolita', 'Greenwich Village', 'Tribeca', 'Boerum Hill', 'SoHo', 'Red Hook', 'Murray Hill',
                   'DUMBO', 'Cobble Hill', 'Financial District', 'Theater District', 'Battery Park City',
                   'Civic Center', 'Vinegar Hill', 'NoHo', 'Columbia St', 'Flatiron District', 'Neponsit',
                   'Willowbrook', 'Belle Harbor'],
    'other': ['Flatbush', 'Bensonhurst', 'Gravesend', 'Shore Acres', 'Sunset Park', 'Co-op City', 'Woodside',
              'Inwood', 'Tompkinsville', 'Tottenville', 'Concord', 'Jackson Heights', 'East Flatbush',
              'Longwood', 'Flatlands', 'Huguenot', 'St. George', 'Bay Ridge', 'Elmhurst', 'Randall Manor',
              'Borough Park', 'Clifton', 'West Brighton', 'Jamaica Estates', 'Kew Gardens', 'Hunts Point',
              'Fort Hamilton', 'Great Kills', 'Bronxdale', 'Corona', 'Castleton Corners', 'Brighton Beach',
              'Claremont Village', 'Highbridge', 'South Beach', 'Pelham Gardens', 'Dyker Heights', 'Arrochar',
              'Morris Park', 'Fordham', 'Coney Island', 'Edenwald', 'Bath Beach', 'Stapleton',
              'Mariners Harbor', 'Port Richmond', 'Midland Beach', 'New Dorp Beach', 'Rosebank',
              'Arden Heights', 'Grant City', 'New Springville', 'Emerson Hill', "Bull's Head", 'Silver Lake',
              'Fieldston', 'Bayswater', 'Sea Gate', 'Westerleigh', 'Graniteville', 'Chelsea, Staten Island',
              'Eltingville', 'Woodrow', 'Rossville', 'Todt Hill']
}

# 역매핑: 동네 -> 클러스터 코드
inv_cluster_map = {neigh: grp for grp, lst in cluster_map.items() for neigh in lst}

borough_map = {
    "Manhattan": [
        # from nbr_grp_04
        "Hell's Kitchen", "Chelsea", "Lower East Side", "East Village",
        "Upper East Side", "Upper West Side", "Chinatown", "Gramercy",
        "Little Italy", "Roosevelt Island",
        # from nbr_grp_03
        "Two Bridges", "East Harlem",
        # from nbr_grp_05
        "Harlem", "Washington Heights", "Maspeth",  # Maspeth 경계상 퀸즈와 접하지만 ManhattanCB5에 일부 포함
        "Morningside Heights",
        # from nbr_grp_01
        "Midtown", "West Village", "Kips Bay", "Nolita",
        "Greenwich Village", "Tribeca", "SoHo", "Murray Hill",
        "Financial District", "Theater District", "Battery Park City",
        "Civic Center", "NoHo", "Flatiron District"
    ],
    "Brooklyn": [
        # nbr_grp_04
        "Prospect Heights", "Williamsburg", "Fort Greene", "Clinton Hill",
        "Gowanus", "Park Slope", "South Slope", "Windsor Terrace",
        "Prospect-Lefferts Gardens", "Downtown Brooklyn",
        "Mill Basin", "Bergen Beach", "Navy Yard", "Gerritsen Beach",
        # nbr_grp_03
        "Bedford-Stuyvesant", "Crown Heights", "Bushwick", "Sheepshead Bay",
        "East New York", "Cypress Hills",
        # nbr_grp_05
        "Bushwick",  # 중복 제거 전후
        # nbr_grp_01
        "Carroll Gardens", "Brooklyn Heights", "Boerum Hill",
        "Red Hook", "DUMBO", "Cobble Hill", "Vinegar Hill", "Columbia St"
    ],
    "Queens": [
        # nbr_grp_04
        "Middle Village", "Long Island City", "Springfield Gardens",
        # nbr_grp_05
        "Astoria", "Ridgewood", "Sunnyside", "Ditmars Steinway",
        "Forest Hills", "Flushing", "Rego Park", "Briarwood",
        "Fresh Meadows", "Holliswood", "Jamaica", "Richmond Hill",
        "Soundview", "Bay Terrace", "College Point", "Little Neck",
        "Ozone Park", "Woodhaven", "St. Albans", "Kew Gardens Hills",
        "Cambria Heights", "Laurelton", "Rosedale", "Arverne",
        "Bayside", "Edgemere", "Far Rockaway", "Neponsit", "Rockaway Park",
        "Bayswater", "Belle Harbor"  # Queens CB14
    ],
    "Bronx": [
        # nbr_grp_04
        "Spuyten Duyvil", "Pelham Bay", "East Morrisania", "University Heights",
        "West Farms",
        # nbr_grp_03
        "Mott Haven", "Eastchester", "Port Morris", "City Island",
        "Bedford-Stuyvesant"  # 경계상 일부 Bronx CB1 포함
        # nbr_grp_05
        "Clason Point", "Kingsbridge", "Allerton", "Norwood",
        "Concourse", "Soundview", "Mount Hope", "Concourse Village",
        "Baychester", "Wakefield", "Mount Eden", "Morrisania",
        "Marble Hill", "Melrose", "Throgs Neck", "Parkchester",
        "Schuylerville", "Belmont", "Morris Heights"
    ],
    "Staten Island": [
        # nbr_grp_04
        "Lighthouse Hill", "New Brighton", "Prince's Bay", "Oakwood",
        "Dongan Hills", "Grymes Hill",
        # nbr_grp_01
        "Willowbrook",
        # other
        "Arrochar", "Annadale", "Arden Heights", "Bay Terrace",
        "Bloomfield", "Bulls Head", "Castleton Corners", "Clifton",
        "Concord", "Eltingville", "Emerson Hill", "Fort Wadsworth",
        "Grant City", "Grasmere", "Great Kills", "Huguenot",
        "Mariners Harbor", "Meiers Corners", "Midland Beach",
        "New Dorp Beach", "New Springville", "Oakwood", "Ocean Breeze",
        "Old Town", "Port Richmond", "Randall Manor", "Rosebank",
        "Seaview", "Shore Acres", "South Beach", "Stapleton",
        "St. George", "Todt Hill", "Tottenville", "West Brighton",
        "Westerleigh", "Woodrow"
    ]
}
# 매핑: 어매니티 구분
amenity_map={
 'common':['Carbon monoxide alarm','Essentials','Hangers','Smoke alarm','Wifi'],
 'high':['Air conditioning','Building staff','Elevator','Gym','Heating','Paid parking off premises','Shampoo'],
 'low-mid':['Cleaning products','Dining table','Exterior security cameras on property','Free street parking','Freezer','Laundromat nearby','Lock on bedroom door','Microwave'],
 'mid':['Cooking basics','Kitchen','Oven'],
 'upper-mid':['Bathtub','Cleaning products','Cooking basics','Dishes and silverware','Elevator','Freezer']
}

# 매핑: 룸 타입 ordinal -> 문자열
type_map = {
    0: 'Private room',
    1: 'Shared room',
    2: 'Entire home/apt',
    3: 'Hotel room'
}

REP_AMENITIES = [
    # Safety
    "Smoke alarm", "Carbon-monoxide alarm", "Fire extinguisher",
    "First-aid kit", "Exterior cameras",
    # Living
    "Wifi", "Air conditioning", "Heating / Hot water",
    "Essentials", "Bed linens & towels", "Hair-dryer / Iron",
    "Washer", "Dryer", "Dedicated workspace", "Pets allowed",
    # Kitchen
    "Kitchen", "Cooking basics", "Refrigerator", "Microwave",
    "Oven", "Stove", "Dishwasher", "Coffee maker",
    # Entertainment
    "TV", "Streaming services", "Sound system / Bluetooth speaker",
    "Board & video games",
    # Outdoor / Facilities
    "Backyard", "Patio / Balcony", "Outdoor furniture", "BBQ grill",
    "Pool", "Bathtub", "Gym", "Free parking", "Paid parking",
    "EV charger", "Elevator"
]

VAL_RMSE_USD = 48.36            # 검증 RMSE
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
                            # 가격 예측
                            price = predict_price(row)
                            # occupancy 예측용 입력값 생성 (공통 입력값만 반영, 나머지는 median/mode)
                            occ_defaults = occ_df[occ_cols].median(numeric_only=True).to_dict()
                            occ_defaults.update(occ_df[occ_cols].mode().iloc[0].to_dict())
                            occ_input = occ_defaults.copy()
                            # 주요 입력값 매핑 (필요시 추가)
                            occ_input['accommodates'] = accommodates
                            occ_input['host_is_superhost'] = 0
                            # room_new_type_encoded, neighbourhood_cluster 등은 필요시 매핑 추가
                            # (여기서는 예시로만 반영, 실제 매핑 필요)
                            occ_input['room_new_type_encoded'] = row.get('room_new_type_ord', 0)
                            occ_input['amenities_cnt'] = row.get('amenities_cnt', occ_input.get('amenities_cnt', 0))
                            # 연간 예약일수 예측
                            occ_days = predict_occupancy(occ_input)
                            # 연수익 계산
                            annual_revenue = price * occ_days
                            recs.append({
                                'Borough'  : bor,
                                'Cluster'  : cl,
                                'AmenGrp'  : amen_grp,
                                'NewGrp'   : new_ord,
                                'Pred'     : price,
                                'OccDays'  : occ_days,
                                'AnnualRev': annual_revenue
                            })
            rec_df = pd.DataFrame(recs)
            near   = rec_df.loc[rec_df['Pred'].sub(target_price).abs() <= 50]
            show   = near if not near.empty else rec_df.iloc[
                        rec_df['Pred'].sub(target_price).abs().sort_values().index[:10]]

            show['Cluster'] = show['Cluster'].map(lambda c: ','.join(cluster_map.get(c,[c])))
            show['NewGrp']  = show['NewGrp'].map(lambda o: ','.join(room_map.get(o,[str(o)])))
            show['Pred']    = show['Pred'].map(lambda x: f"${x:,.0f}")
            show['OccDays'] = show['OccDays'].map(lambda x: f"{x:,.0f}일")
            show['AnnualRev'] = show['AnnualRev'].map(lambda x: f"${x:,.0f}")
            st.subheader("📋 추천 조합 (목표가 ±$50)")
            st.table(show.rename(columns={'Borough':'자치구','Cluster':'지역 클러스터',
                                          'AmenGrp':'Amenity 그룹','NewGrp':'신규 룸그룹',
                                          'Pred':'예측 1박 요금','OccDays':'예상 연간 예약일수','AnnualRev':'예상 연수익'}))

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
    default_opts = [a for a in REP_AMENITIES
                    if clean(a) in [clean(x) for x in amenity_map[grp_label]]]
    sel_am = st.multiselect("주요 Amenity", REP_AMENITIES, default_opts,
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

            # ───────────────────────────────────────────────
            # 슈퍼호스트 숙소 정보 표 추가 (연간수익 Top 10)
            st.markdown("---")
            st.subheader("해당 동네 슈퍼호스트 숙소 정보 (Top 10 연간수익)")
            df_super = occ_df[
                (occ_df['neighbourhood_group_cleansed'] == profile['neighbourhood_group_cleansed']) &
                (occ_df['neigh_cluster_reduced'] == profile['neigh_cluster_reduced']) &
                (occ_df['host_is_superhost'] == 1) &
                (occ_df['is_activate'] == 1)
            ].copy()
            if 'log_price' in df_super.columns:
                df_super['1박당 가격'] = np.expm1(df_super['log_price']).round(0).astype(int)
            if 'estimated_occupancy_l365d' in df_super.columns and '1박당 가격' in df_super.columns:
                df_super['연간수익'] = (df_super['1박당 가격'] * df_super['estimated_occupancy_l365d']).round(0).astype(int)
            show_cols = ['1박당 가격', 'beds', 'amenities_cnt', 'tourism_count', 'infrastructure_count', 'amenity_group', '연간수익']
            show_cols = [col for col in show_cols if col in df_super.columns]
            if show_cols and not df_super.empty:
                df_super = df_super.sort_values('연간수익', ascending=False).head(10)
                st.markdown("<div style='display:flex; justify-content:center;'>", unsafe_allow_html=True)
                st.table(df_super[show_cols].reset_index(drop=True))
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("해당 동네의 슈퍼호스트 숙소 정보가 없습니다.")

# ───────────────────────────────────────────────────────────────
