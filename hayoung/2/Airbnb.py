import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. 데이터 불러오기
df = pd.read_csv('/hye_project/for_machine_learning_2.csv')

# 2. 목표변수와 설명변수 설정
TARGET = 'host_is_superhost'
strategy_cols = [
    'amenities_cnt', 'availability_365', 'price', 'host_about_length_group', 'room_type',
    'name_length_group', 'description_length_group', 'host_has_profile_pic', 
    'host_response_time_score', 'type_amenity_score', 'common_amenity_score', 
    'host_acceptance_rate_score', 'host_identity_verified', 'is_long_term', 'accommodates'
]

X = df[strategy_cols]
y = df[TARGET].astype(int)

# 3. 원핫인코딩
X_encoded = pd.get_dummies(X, drop_first=False)

# 4. 학습/테스트 분할 (stratify로 클래스 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# 5. 랜덤포레스트 모델 정의 및 학습
rf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=30,
    min_samples_split=15,
    min_samples_leaf=10,
    random_state=42,
    class_weight='balanced'
)

rf.fit(X_train, y_train)

# 6. 테스트셋 예측 및 평가
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

print("\n=== 랜덤포레스트 전략모델 성능 평가 ===")
print(classification_report(y_test, y_pred))
print("AUC:", round(roc_auc_score(y_test, y_proba), 4))

# 7. 변수 중요도 출력
importances = pd.Series(rf.feature_importances_, index=X_encoded.columns)
print("\n=== 변수 중요도 ===")
print(importances.sort_values(ascending=False).round(3))

# 8. 학습 시 사용한 컬럼명과 모델 저장
joblib.dump(X_encoded.columns.tolist(), 'train_columns.pkl')  # 컬럼 순서 저장
joblib.dump(rf, 'superhost_rf_model.pkl')  # 모델 저장
print("\n모델과 컬럼 리스트를 저장했습니다.")