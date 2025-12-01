# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pickle

app = FastAPI(title="Mobile Price Regressor API")

# تحميل الموديل المحفوظ (ModelForFastAPI instance)
with open("best_model.pkl", "rb") as f:
    model_wrapper = pickle.load(f)

# --- Pydantic model ديناميكي (مفيد لو حابب) ---
# إذا تعرف أعمدةك استخدمهم هنا. كمثال سأضع بعض الحقول الأساسية:
class SingleInput(BaseModel):
    battery_power: float
    blue: int
    clock_speed: float
    dual_sim: int
    fc: float
    four_g: int
    int_memory: float
    m_dep: float
    mobile_wt: float
    n_cores: int
    pc: float
    px_height: int
    px_width: int
    ram: int
    sc_h: float
    sc_w: float
    talk_time: float
    three_g: int
    touch_screen: int
    wifi: int
    # يمكنك جعل الحقول Optional إذا تريد قبول غيابها


class BatchInput(BaseModel):
    data: List[SingleInput]


# --- نهاية Pydantic model ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Mobile Price Regressor API!"}



@app.post("/predict")
def predict(data: dict):
    pred = model_wrapper.predict_json(data)
    return {"prediction": pred[0]}
