import os
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.preprocess import TextPreprocessor

app = FastAPI()

MODEL_PATH = os.getenv("MODEL_PATH", "models/sentiment_model.pkl")
FE_PATH = os.getenv("FE_PATH", "models/feature_extractor.pkl")

# Load model and feature extractor once at startup
def load_model_and_extractor():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(FE_PATH, 'rb') as f:
        feature_extractor = pickle.load(f)
    return model, feature_extractor

model, feature_extractor = load_model_and_extractor()
preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(request: TextRequest):
    try:
        cleaned_text = preprocessor.clean_text(request.text)
        features = feature_extractor.transform([cleaned_text])
        prediction = int(model.predict(features)[0])
        sentiment = "positive" if prediction == 1 else "negative"
        response = {"sentiment": sentiment, "prediction": prediction}
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            response["confidence"] = float(max(proba))
            response["probabilities"] = {
                "negative": float(proba[0]),
                "positive": float(proba[1])
            }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
