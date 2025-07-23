import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# 1) 매핑 딕셔너리
# ──────────────────────────────────────────────────────────────────────────────
# 클러스터 코드 → 동네 목록 (전략 추천용)
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

# 행정구역(자치구) → 동네 목록 (UI 필터링용)
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

# 룸 타입 코드 → 레이블
type_map = {
    0: 'Private room',
    1: 'Shared room',
    2: 'Entire home/apt',
    3: 'Hotel room'
}

# 신규 룸그룹 코드 → 그룹명
room_group_map = {
    0: 'low-mid',
    1: 'mid',
    2: 'upper-mid',
    3: 'high'
}

# amenity 그룹 코드 → amenity 리스트
amen_group_map={
     'common':['Carbon monoxide alarm','Essentials','Hangers','Smoke alarm','Wifi'],
     'high':['Air conditioning','Building staff','Elevator','Gym','Heating','Paid parking off premises','Shampoo'],
     'low-mid':['Cleaning products','Dining table','Exterior security cameras on property','Free street parking','Freezer','Laundromat nearby','Lock on bedroom door','Microwave'],
     'mid':['Cooking basics','Kitchen','Oven'],
     'upper-mid':['Bathtub','Cleaning products','Cooking basics','Dishes and silverware','Elevator','Freezer']
}

# 화면에 보여줄 전체 amenity 리스트
amen_selection_map = [
    # Safety
    "Smoke alarm", "Carbon-monoxide alarm", "Fire extinguisher",
    "First-aid kit", "Exterior cameras",
    # Living
    "Wifi", "Air conditioning", "Heating", "Hot water",
    "Shampoo", "Conditioner","Shower gel", "Body soap", "Bed linens", "towels", "Hair-dryer", "Iron",
    "Washer", "Dryer", "Dedicated workspace", "Pets allowed", "Clothing storage"
    # Kitchen
    "Kitchen", "Cooking basics", "Refrigerator", "Microwave",
    "Oven", "Stove", "Dishwasher", "Coffee maker","Wine glasses", "Toaster", "Dining table",
    # Entertainment
    "TV", "Streaming services", "Sound system / Bluetooth speaker",
    "Board & video games",
    # Outdoor / Facilities
    "Backyard", "Patio / Balcony", "Outdoor furniture", "BBQ grill",
    "Pool", "Bathtub", "Gym", "Free parking", "Paid parking",
    "EV charger", "Elevator", "Lockbox", "Pets allowed", "Self check-in"
]


# ──────────────────────────────────────────────────────────────────────────────
#  카테고리별 컬럼 그룹 (모델 입력용 기존 컬럼명)
# ──────────────────────────────────────────────────────────────────────────────
cat_cols = [
    'neigh_cluster_reduced', 'neighbourhood_group_cleansed',
    'room_type_ord', 'room_new_type_ord', 'room_structure_type', 'amen_grp',
    'description_length_group', 'name_length_group'
]
num_cols = [
    'latitude', 'longitude', 'accommodates', 'bath_score_mul',
    'amenities_cnt', 'review_scores_rating',
    'number_of_reviews', 'number_of_reviews_ltm', 'region_score_norm',
    'host_response_time_score', 'host_response_rate_score'
]
bin_cols = [
    'instant_bookable', 'is_long_term', 'host_is_superhost',
    'has_Air_conditioning', 'has_Wifi', 'has_Bathtub',
    'has_Carbon_monoxide_alarm', 'has_Elevator',
    'neighborhood_overview_exists'
]
other_flags = ['grp01_high', 'grp04_high']

# 모델이 기대하는 원본 피처 리스트
model_features_old = cat_cols + num_cols + bin_cols + other_flags

# ──────────────────────────────────────────────────────────────────────────────
#   CSV에 저장된 새 컬럼명 매핑 (원본→새 이름)
# ──────────────────────────────────────────────────────────────────────────────
rename_old2new = {
    'neigh_cluster_reduced':        'nei_cluster_code',
    'cluster_label':                'nei_cluster_label',
    'grp01_high':                   'nei_cluster_grp01_high',
    'grp04_high':                   'nei_cluster_grp04_high',
    'neighbourhood_group_cleansed': 'nei_borough',
    'neighbourhood_cleansed':       'nei_neighbourhood',
    'room_type_ord':                'room_type_code',
    'room_type_label':              'room_type_label',
    'room_structure_type':          'room_structure',
    'room_new_type_ord':            'room_group_code',
    'room_group_label':             'room_group_label',
    'description_length_group':     'info_des_len',
    'name_length_group':            'info_name_len',
    'neighborhood_overview_exists': 'info_overview_is',
    'instant_bookable':             'info_instant_bookable',
    'is_long_term':                 'info_long_term_allowed',
    'host_response_time_score':     'host_resp_time',
    'host_response_rate_score':     'host_resp_rate',
    'host_is_superhost':            'host_is_superhost',
    'has_Air_conditioning':         'amen_has_air_conditioning',
    'has_Wifi':                     'amen_has_wifi',
    'has_Bathtub':                  'amen_has_bathtub',
    'has_Carbon_monoxide_alarm':    'amen_has_carbon_alarm',
    'has_Elevator':                 'amen_has_elevator',
    'amenities_cnt':                'amen_count',
    'amen_grp':                     'amenity_group',
    'review_scores_rating':         'review_score',
    'region_score_norm':            'region_score_norm',
    'number_of_reviews':            'review_num_total',
    'number_of_reviews_ltm':        'review_num_30d',
    'accommodates':                 'facility_accommodates',
    'bath_score_mul':               'facility_bath_score'
}
# 역매핑: 새 이름 → 원본 컬럼명
rename_new2old = {new: old for old, new in rename_old2new.items()}

# ──────────────────────────────────────────────────────────────────────────────
#  UI에서 사용할 컬럼 그룹 (새 컬럼명)
# ──────────────────────────────────────────────────────────────────────────────

def get_column_groups():
    """
    Returns dictionary of column group lists under final CSV (new) names:
      - index_cols: ['id','host_id']
      - model_features_new: renamed model inputs
      - mapping_cols: UI-only labels
      - ui_features: all columns to load in app
    """
    index_cols = ['id', 'host_id']
    model_features_new = [rename_old2new.get(col, col) for col in model_features_old]
    mapping_cols = [
        rename_old2new['cluster_label'],
        rename_old2new['room_type_label'],
        rename_old2new['room_group_label']
    ]
    ui_features = index_cols + model_features_new + mapping_cols
    return {
        'index_cols': index_cols,
        'model_features_new': model_features_new,
        'mapping_cols': mapping_cols,
        'ui_features': ui_features
    }

# ──────────────────────────────────────────────────────────────────────────────
#  Helper: 새 컬럼명 dict → 원본 컬럼명 dict
# ──────────────────────────────────────────────────────────────────────────────

def row_new_to_old(row_new: dict) -> dict:
    """
    Convert a record's keys from renamed CSV names back to original model feature names.
    """
    return {rename_new2old.get(k, k): v for k, v in row_new.items()}
