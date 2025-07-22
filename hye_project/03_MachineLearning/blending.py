
import numpy as np
import joblib
from catboost import CatBoostRegressor

# 모델 로드
cat = CatBoostRegressor()
cat.load_model('catboost_tuned_model.cbm')
hgb = joblib.load('histgb_model.joblib')

def predict_price(df_new):
    # df_new 는 features_final 컬럼을 가진 DataFrame
    pred_cat = cat.predict(df_new)
    pred_hgb = hgb.predict(df_new)
    pred_log_blend = (pred_cat + pred_hgb) / 2
    return np.expm1(pred_log_blend)
