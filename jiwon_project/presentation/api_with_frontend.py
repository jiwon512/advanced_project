from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# 1) 정적 파일들을 serve 할 디렉터리 마운트
#    "/" 경로로 들어오는 요청 중 파일이 있으면 frontend/ 폴더에서 찾아서 응답합니다.
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

# 2) (예시) Pydantic 모델 정의
class ListingFeatures(BaseModel):
    price: float
    amenities_cnt: int
    availability_365: int
    instant_bookable: int
    # 나머지 필요한 필드...

# 3) 학습된 전처리기 & 스태킹 모델 불러오기
preprocessor = joblib.load("models/preprocessor.joblib")
stack_model = joblib.load("models/stacking_model.joblib")

# 4) API 엔드포인트: 수익 예측
@app.post("/predict_revenue")
def predict_revenue(feat: ListingFeatures):
    df_in = pd.DataFrame([feat.dict()])
    Xp = preprocessor.transform(df_in)
    reserved_days = stack_model.predict(Xp)[0]
    revenue = reserved_days * feat.price
    return {
        "predicted_reserved_days": float(reserved_days),
        "predicted_annual_revenue": float(revenue)
    }

# 필요하다면 다른 엔드포인트도 여기에 추가
