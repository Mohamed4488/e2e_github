from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from infer import predict_survival


app = FastAPI()

class Features(BaseModel):
    features:List[float]
    
@app.post("/predict")
async def predict(features: Features):cl
    prediction = predict_survival(features.features)
    return {"prediction": prediction}