import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# 0) Use cols
# ──────────────────────────────────────────────────────────────────────────────

index_cols = ['id','host_id']
y_cols = ['price', 'log_price']

# - 혜영 컬럼 --------------------------------------------------------------------
hye_cols = [
    'latitude',
    'longitude',
    'neigh_cluster_reduced',
    'grp01_high',
    'grp04_high',

    'neighbourhood_group_cleansed',
    'neighbourhood_cleansed',

    'room_type_ord',
    'room_structure_type',
    'room_new_type_ord',
    'accommodates',
    'bath_score_mul',

    'has_Air_conditioning',
    'has_Wifi',
    'has_Bathtub',
    'has_Carbon_monoxide_alarm',
    'has_Elevator',
    'amenities_cnt',
    'amen_grp',

    'description_length_group',
    'name_length_group',
    'neighborhood_overview_exists',
    'instant_bookable',
    'is_long_term',

    'host_response_time_score',
    'host_response_rate_score',
    'host_is_superhost',

    'review_scores_rating',
    'region_score_norm',
    'number_of_reviews',
    'number_of_reviews_ltm'
]

all_cols = index_cols + y_cols + hye_cols

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
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # 1) 필요한 컬럼만 복사
    df_out = df[all_cols].copy()

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


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    비슷한 의미의 컬럼들을 일관된 스네이크 케이스 네이밍으로 통합합니다.
    """
    rename_map = {
        # 클러스터
        'neigh_cluster_reduced':        'nei_cluster_code',
        'cluster_label':                'nei_cluster_label',  # UI용 매핑 컬럼
        # 클러스터-가격 플래그
        'grp01_high':                   'nei_cluster_grp01_high',
        'grp04_high':                   'nei_cluster_grp04_high',
        # 자치구·동네
        'neighbourhood_group_cleansed': 'nei_borough',
        'neighbourhood_cleansed':       'nei_neighbourhood',
        # 룸 타입
        'room_type_ord':                'room_type_code',
        'room_type_label':              'room_type_label',       # UI용 매핑 컬럼
        # 구조 그룹
        'room_structure_type':          'room_structure',
        'room_new_type_ord':            'room_group_code',
        'room_group_label':             'room_group_label',      # UI용 매핑 컬럼
        # 설명란 존재 여부
        'description_length_group':     'info_des_len',
        'name_length_group':            'info_name_len',
        'neighborhood_overview_exists': 'info_overview_is',
        'instant_bookable':             'info_instant_bookable',
        'is_long_term':                 'info_long_term_allowed',
        # 호스트 응답
        'host_response_time_score':     'host_resp_time',
        'host_response_rate_score':     'host_resp_rate',
        'host_is_superhost':            'host_is_superhost',
        # 어매니티
        'has_Air_conditioning':         'amen_has_air_conditioning',
        'has_Wifi':                     'amen_has_wifi',
        'has_Bathtub':                  'amen_has_bathtub',
        'has_Carbon_monoxide_alarm':    'amen_has_carbon_alarm',
        'has_Elevator':                 'amen_has_elevator',
        'amenities_cnt':                'amen_count',
        'amen_grp':                     'amenity_group',
        # 리뷰 수치형
        'review_scores_rating':         'review_score',
        'region_score_norm':            'region_score_norm',
        'number_of_reviews':            'review_num_total',
        'number_of_reviews_ltm':        'review_num_30d',
        'accommodates':                 'facility_accommodates',
        'bath_score_mul':               'facility_bath_score'
    }
    return df.rename(columns=rename_map)

# ──────────────────────────────────────────────────────────────────────────────
# 3) 스크립트 실행부
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 중간 CSV 읽기
    df_raw = pd.read_csv("/Users/hyeom/Documents/GitHub/advanced_project/hye_project/for_machine_learning_2.csv")
    # 추가 전처리
    df_final = preprocess(df_raw)
    #df_final = rename_columns(df_final)
    # 최종 저장
    df_final.to_csv("/Users/hyeom/Documents/GitHub/advanced_project/hye_project/04_app/processed_hye.csv", index=False)
    print("✅ processed_app.csv 생성 완료")

