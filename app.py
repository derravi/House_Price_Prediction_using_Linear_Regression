from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pickle
import pandas as pd
from Schema.pydantic_model import UserInput


with open('model/model.pkl','rb') as f:
    model = pickle.load(f)

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to the House Price Prediction API!",
            "this and add into you browser after Host_ip for Open House Price Prediction Apis:":"/docs"} 

@app.post("/predict")
def predict_premium(data:UserInput):
    df = pd.DataFrame([{
        "Area_sqft": data.Area_sqft,
        "Bedrooms": data.Bedrooms,
        "Bathrooms": data.Bathrooms,
        "Floors": data.Floors,
        "Year_Built": data.Year_Built
    }])

    prediction = model.predict(df)[0][0]

    return JSONResponse(status_code=200,content={"predicted_price": float(round(prediction, 2)),
                                                 "message": f"The estimated price of the house is ₹{round(prediction,2)}"})

