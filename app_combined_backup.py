import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

# ====== 1. 경로/피처리스트/데이터 로드 ======
# 가격 예측 모델
PRICE_MODEL_PATH = "/Users/Jiwon/Documents/GitHub/advanced_project/jiwon_price.pkl"
PRICE_DF_PATH = "/Users/Jiwon/Documents/GitHub/advanced_project/jiwon_project/4.app/backup/processed_hye.csv"

# occupancy 예측 모델
OCC_MODEL_PATH = "/Users/Jiwon/Documents/GitHub/advanced_project/occupancy_voting_model.pkl"
OCC_DF_PATH = "/Users/Jiwon/Documents/GitHub/advanced_project/jiwon_project/presentation/jiwon_entire.csv"

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
    return pd.read_csv(path)
@st.cache_resource
def load_model(path):
    return joblib.load(path)
price_df = load_df(PRICE_DF_PATH)
occ_df = load_df(OCC_DF_PATH)
price_model = load_model(PRICE_MODEL_PATH)
occ_model = load_model(OCC_MODEL_PATH)

# id 기준 merge (중복 컬럼 하나만 남기고, inner join)
if 'id' in price_df.columns and 'id' in occ_df.columns:
    merged_df = pd.merge(price_df, occ_df, on='id', how='inner', suffixes=('_price', '_occ'))
    # 중복 컬럼(양쪽에 다 있는 컬럼) 리스트 추출
    common_cols = [col for col in price_df.columns if col in occ_df.columns and col != 'id']
    for col in common_cols:
        occ_col = f'{col}_occ'
        price_col = f'{col}_price'
        # price_df 기준으로 남기고, _occ는 삭제
        if occ_col in merged_df.columns:
            merged_df.drop(columns=[occ_col], inplace=True)
        if price_col in merged_df.columns:
            merged_df.rename(columns={price_col: col}, inplace=True)
else:
    # id 컬럼명이 다르면 아래처럼 수정 (예시)
    # merged_df = pd.merge(price_df, occ_df, left_on='listing_id', right_on='id', how='inner', suffixes=('_price', '_occ'))
    merged_df = None  # 에러 방지용


# ====== [backup_app.py에서 가져온 주요 상수/매핑/함수] ======
# (이미 정의된 것은 중복 정의하지 않음)

# 매핑: 신규 룸 타입 -> 구조 리스트
room_map = {
    1: ['rental unit', 'guest suite', 'place', 'townhouse', 'serviced apartment', 'guesthouse'],
    2: ['condo', 'loft', 'houseboat', 'boutique hotel', 'boat', 'villa', 'tiny home', 'bungalow',
                  'cottage', 'aparthotel', 'barn'],
    0: ['home', 'bed and breakfast', 'casa particular', 'vacation home', 'earthen home', 'camper/rv',
                'hostel', 'kezhan', 'ranch', 'religious building', 'dome'],
    3: ['hotel', 'resort', 'tower']
}
# 매핑: 클러스터 코드 -> 동네 목록
cluster_map = {
    'nbr_grp_04': ["Prospect Heights", "Williamsburg", "Hell's Kitchen", "Fort Greene", "Clinton Hill",
                   "Chelsea", "Gowanus", "Lower East Side", "East Village", "Park Slope", "Upper East Side",
                   "Middle Village", "South Slope", "Upper West Side", "Chinatown", "Windsor Terrace",
                   "Prospect-Lefferts Gardens", "Downtown Brooklyn", "Long Island City", "Spuyten Duyvil",
                   "Gramercy", "Lighthouse Hill", "Springfield Gardens", "Little Italy", "New Brighton",
                   "Howland Hook", "Roosevelt Island", "Pelham Bay", "East Morrisania", "Mill Basin",
                   "Bergen Beach", "Prince's Bay", "Navy Yard", "Gerritsen Beach", "Breezy Point",
                   "University Heights", "West Farms", "Oakwood", "Dongan Hills", "Grymes Hill"],
    'nbr_grp_03': ["East Harlem", "Bedford-Stuyvesant", "Crown Heights", "Mott Haven", "Morningside Heights",
                   "Rockaway Beach", "Eastchester", "Sheepshead Bay", "East New York", "Two Bridges",
                   "City Island", "Port Morris", "Arverne", "Queens Village", "Canarsie", "Bay Terrace",
                   "Forest Hills", "Unionport", "Jamaica", "Bayside", "South Ozone Park", "Howard Beach",
                   "Fresh Meadows", "Bellerose", "Edgemere", "Stuyvesant Town", "Rosedale", "Kew Gardens Hills",
                   "Laurelton", "Tremont", "Olinville", "College Point", "Westchester Square",
                   "North Riverdale", "Douglaston", "Far Rockaway", "Cambria Heights", "Jamaica Hills",
                   "Woodlawn", "Castle Hill", "Van Nest", "Country Club", "Riverdale"],
    'nbr_grp_05': ["Harlem", "Washington Heights", "Ditmars Steinway", "Astoria", "Ridgewood", "Clason Point",
                   "Kingsbridge", "Bushwick", "Sunnyside", "Kensington", "Briarwood", "Allerton", "Flushing",
                   "East Elmhurst", "Norwood", "Concourse", "Richmond Hill", "Maspeth", "Soundview",
                   "Rego Park", "Woodhaven", "Mount Hope", "Concourse Village", "Midwood", "Ozone Park",
                   "Cypress Hills", "Manhattan Beach", "Brownsville", "Holliswood", "Baychester", "Wakefield",
                   "St. Albans", "Whitestone", "Mount Eden", "Glendale", "Morrisania", "Marble Hill", "Hollis",
                   "Williamsbridge", "Melrose", "Throgs Neck", "Parkchester", "Schuylerville", "Belmont",
                   "Morris Heights", "Little Neck"],
    'nbr_grp_01': ["Carroll Gardens", "Midtown", "Greenpoint", "West Village", "Brooklyn Heights", "Kips Bay",
                   "Nolita", "Greenwich Village", "Tribeca", "Boerum Hill", "SoHo", "Red Hook", "Murray Hill",
                   "DUMBO", "Cobble Hill", "Financial District", "Theater District", "Battery Park City",
                   "Civic Center", "Vinegar Hill", "NoHo", "Columbia St", "Flatiron District", "Neponsit",
                   "Willowbrook", "Belle Harbor"],
    'other': ["Flatbush", "Bensonhurst", "Gravesend", "Shore Acres", "Sunset Park", "Co-op City", "Woodside",
              "Inwood", "Tompkinsville", "Tottenville", "Concord", "Jackson Heights", "East Flatbush",
              "Longwood", "Flatlands", "Huguenot", "St. George", "Bay Ridge", "Elmhurst", "Randall Manor",
              "Borough Park", "Clifton", "West Brighton", "Jamaica Estates", "Kew Gardens", "Hunts Point",
              "Fort Hamilton", "Great Kills", "Bronxdale", "Corona", "Castleton Corners", "Brighton Beach",
              "Claremont Village", "Highbridge", "South Beach", "Pelham Gardens", "Dyker Heights", "Arrochar",
              "Morris Park", "Fordham", "Coney Island", "Edenwald", "Bath Beach", "Stapleton",
              "Mariners Harbor", "Port Richmond", "Midland Beach", "New Dorp Beach", "Rosebank",
              "Arden Heights", "Grant City", "New Springville", "Emerson Hill", "Bull's Head", "Silver Lake",
              "Fieldston", "Bayswater", "Sea Gate", "Westerleigh", "Graniteville", "Chelsea, Staten Island",
              "Eltingville", "Woodrow", "Rossville", "Todt Hill"]
}
# 역매핑: 동네 -> 클러스터 코드
inv_cluster_map = {neigh: grp for grp, lst in cluster_map.items() for neigh in lst}

borough_map = {
    "Manhattan": [
        "Hell's Kitchen", "Chelsea", "Lower East Side", "East Village",
        "Upper East Side", "Upper West Side", "Chinatown", "Gramercy",
        "Little Italy", "Roosevelt Island", "Two Bridges", "East Harlem",
        "Harlem", "Washington Heights", "Maspeth", "Morningside Heights",
        "Midtown", "West Village", "Kips Bay", "Nolita", "Greenwich Village",
        "Tribeca", "SoHo", "Murray Hill", "Financial District", "Theater District",
        "Battery Park City", "Civic Center", "NoHo", "Flatiron District"
    ],
    "Brooklyn": [
        "Prospect Heights", "Williamsburg", "Fort Greene", "Clinton Hill",
        "Gowanus", "Park Slope", "South Slope", "Windsor Terrace",
        "Prospect-Lefferts Gardens", "Downtown Brooklyn", "Mill Basin", "Bergen Beach",
        "Navy Yard", "Gerritsen Beach", "Bedford-Stuyvesant", "Crown Heights",
        "Bushwick", "Sheepshead Bay", "East New York", "Cypress Hills", "Carroll Gardens",
        "Brooklyn Heights", "Boerum Hill", "Red Hook", "DUMBO", "Cobble Hill",
        "Vinegar Hill", "Columbia St"
    ],
    "Queens": [
        "Middle Village", "Long Island City", "Springfield Gardens", "Astoria", "Ridgewood",
        "Sunnyside", "Ditmars Steinway", "Forest Hills", "Flushing", "Rego Park", "Briarwood",
        "Fresh Meadows", "Holliswood", "Jamaica", "Richmond Hill", "Soundview", "Bay Terrace",
        "College Point", "Little Neck", "Ozone Park", "Woodhaven", "St. Albans", "Kew Gardens Hills",
        "Cambria Heights", "Laurelton", "Rosedale", "Arverne", "Bayside", "Edgemere", "Far Rockaway",
        "Neponsit", "Rockaway Park", "Bayswater", "Belle Harbor"
    ],
    "Bronx": [
        "Spuyten Duyvil", "Pelham Bay", "East Morrisania", "University Heights", "West Farms",
        "Mott Haven", "Eastchester", "Port Morris", "City Island", "Bedford-Stuyvesant",
        "Clason Point", "Kingsbridge", "Allerton", "Norwood", "Concourse", "Soundview",
        "Mount Hope", "Concourse Village", "Baychester", "Wakefield", "Mount Eden", "Morrisania",
        "Marble Hill", "Melrose", "Throgs Neck", "Parkchester", "Schuylerville", "Belmont",
        "Morris Heights"
    ],
    "Staten Island": [
        "Lighthouse Hill", "New Brighton", "Prince's Bay", "Oakwood", "Dongan Hills", "Grymes Hill",
        "Willowbrook", "Arrochar", "Annadale", "Arden Heights", "Bay Terrace", "Bloomfield",
        "Bulls Head", "Castleton Corners", "Clifton", "Concord", "Eltingville", "Emerson Hill",
        "Fort Wadsworth", "Grant City", "Grasmere", "Great Kills", "Huguenot", "Mariners Harbor",
        "Meiers Corners", "Midland Beach", "New Dorp Beach", "New Springville", "Oakwood",
        "Ocean Breeze", "Old Town", "Port Richmond", "Randall Manor", "Rosebank", "Seaview",
        "Shore Acres", "South Beach", "Stapleton", "St. George", "Todt Hill", "Tottenville",
        "West Brighton", "Westerleigh", "Woodrow"
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
    "Smoke alarm", "Carbon-monoxide alarm", "Fire extinguisher",
    "First-aid kit", "Exterior cameras", "Wifi", "Air conditioning", "Heating / Hot water",
    "Essentials", "Bed linens & towels", "Hair-dryer / Iron", "Washer", "Dryer", "Dedicated workspace",
    "Pets allowed", "Kitchen", "Cooking basics", "Refrigerator", "Microwave", "Oven", "Stove",
    "Dishwasher", "Coffee maker", "TV", "Streaming services", "Sound system / Bluetooth speaker",
    "Board & video games", "Backyard", "Patio / Balcony", "Outdoor furniture", "BBQ grill", "Pool",
    "Bathtub", "Gym", "Free parking", "Paid parking", "EV charger", "Elevator"
]
VAL_RMSE_USD = 48.36
MIN_NIGHTLY  = 10.0
MAX_NIGHTLY  = 900.0
max_acc      = int(price_df['accommodates'].max())

defaults = {
    **price_df[num_cols].median().to_dict(),
    **price_df[cat_cols].mode().iloc[0].to_dict(),
    **{c:0 for c in bin_cols},
    **{f:0 for f in other_flags}
}

def add_strategy(bucket: list, label: str, test_row: dict, base: float):
    delta = predict_price(test_row) - base
    bucket.append((label, delta))

# ====== 2. 예측 함수 ======
def predict_price(row: dict) -> float:
    X = pd.DataFrame([row])[hye_features]
    return float(np.expm1(price_model.predict(X)[0]))

def predict_occupancy(row: dict) -> float:
    X = pd.DataFrame([row])[occ_cols].fillna(0)
    return float(occ_model.predict(X)[0])

# ====== 3. UI/UX ======
st.markdown(
    "<h1 style='text-align:center; font-size:2.5em; font-weight:bold;'> NYC Airbnb Host</h1>",
    unsafe_allow_html=True
)
mode = st.radio("", ["예비 호스트", "기존 호스트"])

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
        st.markdown(f"**{bor}'s town**")
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
                    df_super = merged_df[
                        (merged_df['neighbourhood_group_cleansed'] == bor) &
                        (merged_df['neighbourhood_cleansed'] == neigh) &
                        (merged_df['host_is_superhost'] == 1) &
                        (merged_df['is_activate'] == 1)
                    ]
                    # 사용자가 선택한 장단기, 최대 숙박인원, 희망룸타입(ordinal)으로 groupby
                    group_key = {
                        'neighbourhood_group_cleansed': bor,
                        'neighbourhood_cleansed': neigh,
                        'is_long_term': int(is_long_term),
                        'accommodates': accommodates,
                        'room_new_type_encoded': rt_choice  # rt_choice가 실제로 room_new_type_encoded와 매핑되는 값이어야 함
                    }
                    df_group = df_super[
                        (df_super['is_long_term'] == group_key['is_long_term']) &
                        (df_super['accommodates'] == group_key['accommodates']) &
                        (df_super['room_new_type_encoded'] == group_key['room_new_type_encoded'])
                    ]
                    if not df_group.empty:
                        if 'log_price' in df_group.columns:
                            df_group['1박당 가격'] = np.expm1(df_group['log_price']).round(0).astype(int)
                        if 'estimated_occupancy_l365d' in df_group.columns and '1박당 가격' in df_group.columns:
                            df_group['연간수익'] = (df_group['1박당 가격'] * df_group['estimated_occupancy_l365d']).round(0).astype(int)
                        show_cols = ['1박당 가격', 'amenities_cnt', 'tourism_count', 'infrastructure_count', 'amen_grp', '연간수익']
                        show_cols = [col for col in show_cols if col in df_group.columns]
                        df_show = df_group[show_cols].sort_values('연간수익', ascending=False).head(10)
                        # 표 숫자 모두 정수로 변환
                        int_cols = ['1박당 가격', 'amenities_cnt', 'tourism_count', 'infrastructure_count', '연간수익']
                        for col in int_cols:
                            if col in df_show.columns:
                                df_show[col] = df_show[col].astype(int)
                        st.markdown("<div style='display:flex; justify-content:center;'>", unsafe_allow_html=True)
                        st.table(df_show.reset_index(drop=True))
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.info("해당 조건의 슈퍼호스트 숙소 정보가 없습니다.")


# ====== 5. 기존 호스트 ======
if mode == "기존 호스트":
    key_prefix = "old_"
    st.header("📈 기존 호스트를 위한 전략 분석")
    profile = {}

    # ── 5-1) 프로필 입력 ─────────────────────────────
    # 자치구 & 동네
    bor = st.selectbox("자치구(Borough)",
                       price_df['neighbourhood_group_cleansed'].unique(),
                       key=f"{key_prefix}bor")
    profile['neighbourhood_group_cleansed'] = bor

    # borough_map, inv_cluster_map 등은 상단에 정의 필요
    neigh_list   = borough_map.get(bor, [])
    neigh_counts = price_df[price_df['neighbourhood_group_cleansed']==bor]['neighbourhood_cleansed'].value_counts()
    neigh_sorted = sorted(neigh_list, key=lambda x: neigh_counts.get(x,0), reverse=True)
    neigh = st.selectbox("동네(Neighborhood)", neigh_sorted, key=f"{key_prefix}neigh")
    profile['neigh_cluster_reduced'] = inv_cluster_map.get(neigh, 'other')

    # 룸 타입, 구조
    rt_ordinals = sorted(price_df['room_type_ord'].unique())
    rt_labels   = [type_map[o] for o in rt_ordinals]
    rt_lb       = st.selectbox("룸 타입", rt_labels, key=f"{key_prefix}rt")
    profile['room_type_ord'] = rt_ordinals[rt_labels.index(rt_lb)]

    struct_opts = price_df[price_df['room_type_ord']==profile['room_type_ord']]['room_structure_type'].value_counts().index.tolist()
    struct = st.selectbox("숙소 구조(설명란)", struct_opts, key=f"{key_prefix}struct")
    profile['room_structure_type'] = struct
    inv_room = {s:g for g, lst in room_map.items() for s in lst}
    profile['room_new_type_ord'] = inv_room.get(struct,0)

    with st.expander("숙소 설명란은 어떻게 선택하나요? 📋", expanded=False):
        st.image(
            "hye_project/structure_example.png",
            use_column_width=True
        )
        st.write("숙소 설명란의 해당 정보를 바탕으로 작성해 주세요! 임의로 선택시 예측율이 떨어질 수 있어요.")

    # 숙박 인원
    acc = st.number_input("최대 숙박 인원", 1, max_acc,
                          int(defaults['accommodates']), 1,
                          key=f"{key_prefix}acc")
    profile['accommodates'] = acc

    # 예약/정책 토글
    profile['instant_bookable']   = int(st.toggle("Instant Bookable", key=f"{key_prefix}inst"))
    profile['is_long_term']       = int(st.toggle("장기 숙박 허용", key=f"{key_prefix}long"))
    profile['host_is_superhost']  = int(st.toggle("슈퍼호스트 여부", key=f"{key_prefix}super"))

    # Amenity 멀티선택
    def clean(s:str)->str: return re.sub(r'[\uD800-\uDFFF]', '', s).lower().strip()
    grp_label = {0:'low-mid',1:'mid',2:'upper-mid',3:'high'}.get(profile['room_new_type_ord'],'common')
    default_opts = [a for a in REP_AMENITIES if clean(a) in [clean(x) for x in amenity_map[grp_label]]]
    sel_am = st.multiselect("주요 Amenity", REP_AMENITIES, default_opts, key=f"{key_prefix}amen")
    profile['amenities_cnt'] = len(sel_am)
    profile['amen_grp']      = grp_label
    for flag in ['air conditioning','wifi','bathtub','carbon monoxide alarm','elevator']:
        profile[f"has_{flag.replace(' ','_')}"] = int(flag in map(clean, sel_am))

    # ───────────────────────────────────────────────
    # 5-2) 성과/목표 입력
    st.subheader("📊 성과 및 목표 입력")
    booked_days = st.number_input("한 달 예약된 날 수", 1, 31, 20, key=f"{key_prefix}days")
    MIN_REV = booked_days*MIN_NIGHTLY
    MAX_REV = booked_days*MAX_NIGHTLY

    curr_rev = st.number_input("현재 월 수익 ($)", MIN_REV, MAX_REV, 3000.0, 50.0, key=f"{key_prefix}curr")
    desired_rev = st.number_input("목표 월 수익 ($)", MIN_REV, MAX_REV, 4000.0, 50.0, key=f"{key_prefix}goal")

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
            for alt in price_df['amen_grp'].unique():
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

