import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# 0) Use cols
# ──────────────────────────────────────────────────────────────────────────────

index_cols = ['id','host_id']

# - 혜영 컬럼 --------------------------------------------------------------------
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

# ──────────────────────────────────────────────────────────────────────────────
# 1) Mapping Dictionary
# ──────────────────────────────────────────────────────────────────────────────

room_map = {
    1: ['rental unit', 'guest suite', 'place', 'townhouse', 'serviced apartment', 'guesthouse'], #mid
    2: ['condo', 'loft', 'houseboat', 'boutique hotel', 'boat', 'villa', 'tiny home', 'bungalow', #upper-mid
                  'cottage', 'aparthotel', 'barn'],
    0: ['home', 'bed and breakfast', 'casa particular', 'vacation home', 'earthen home', 'camper/rv', #low-mid
                'hostel', 'kezhan', 'ranch', 'religious building', 'dome'],
    3: ['hotel', 'resort', 'tower'] # high
}

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

type_map = {
    0: 'Private room',
    1: 'Shared room',
    2: 'Entire home/apt',
    3: 'Hotel room'
}

# ──────────────────────────────────────────────────────────────────────────────
# 2) 전처리 함수
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# 2) 전처리 함수
# ──────────────────────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # 1) 필요한 컬럼만 복사
    df_out = df[index_cols + features].copy()

    # 2) UI용 매핑 컬럼 생성
    #   a) cluster_label: 코드 → 쉼표로 연결된 동네 목록
    df_out['cluster_label'] = (
        df_out['neigh_cluster_reduced']
          .map(lambda c: ', '.join(cluster_map.get(c, [])))
    )
    #   b) room_type_label: ordinal → 설명 문자열
    df_out['room_type_label'] = df_out['room_type_ord'].map(type_map)
    #   c) room_group_label: 신규 룸그룹 ordinal → 그룹명
    df_out['room_group_label'] = df_out['room_new_type_ord'].map(room_map)

    # 3) 타입 변환
    #   수치형 → float (NaN→0)
    df_out[num_cols] = (
        df_out[num_cols]
          .apply(pd.to_numeric, errors='coerce')
          .fillna(0)
          .astype(float)
    )
    #   범주형 → str (NaN→"missing")
    df_out[cat_cols] = (
        df_out[cat_cols]
          .astype(str)
          .fillna('missing')
    )
    #   바이너리 & 플래그 → int
    df_out[bin_cols + other_flags] = (
        df_out[bin_cols + other_flags]
          .fillna(0)
          .astype(int)
    )

    return df_out

# ──────────────────────────────────────────────────────────────────────────────
# 3) 스크립트 실행부
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 중간 CSV 읽기
    df_raw = pd.read_csv("/Users/hyeom/Documents/GitHub/advanced_project/hye_project/for_machine_learning_2.csv")
    # 추가 전처리
    df_final = preprocess(df_raw)
    # 최종 저장
    df_final.to_csv("/Users/hyeom/Documents/GitHub/advanced_project/hye_project/04_app/preprocessed_hye.csv", index=False)
    print("✅ processed_final.csv 생성 완료")