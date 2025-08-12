from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# Load model using pickle
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define class names for predictions
class_names = ["setosa", "versicolor", "virginica"]

# Initialize FastAPI app
app = FastAPI(title="Iris Classifier API with Pickle")

# Define input schema
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define prediction route
@app.post("/predict")
def predict_species(features: IrisFeatures):
    input_data = np.array([[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]])
    prediction = model.predict(input_data)[0]
    predicted_class = class_names[prediction]
    return {"predicted_class": predicted_class}