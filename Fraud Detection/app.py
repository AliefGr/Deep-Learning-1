from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from pydantic import BaseModel, Field
from datetime import datetime

app = FastAPI(
    title="Fraud Detection API",
    description="API untuk deteksi fraud/kecurangan menggunakan KNN model dan TF-IDF",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Izinkan akses dari React (localhost:5173) 
app.add_middleware( CORSMiddleware, 
    allow_origins=["http://localhost:5173"], # bisa juga ["*"] untuk semua origin 
    allow_credentials=True, 
    allow_methods=["*"], allow_headers=["*"], 
)

model = None
tfidf_vectorizer = None
model_loaded = False

def load_models():
    """
    Load models lazily when first request comes
    """
    global model, tfidf_vectorizer, model_loaded
    
    if model_loaded:
        return True
    
    try:
        import joblib
        
        # Check if model files exist
        if not os.path.exists('knn_model.pkl'):
            raise FileNotFoundError("knn_model.pkl not found!")
        
        if not os.path.exists('tfidf_vectorizer.pkl'):
            raise FileNotFoundError("tfidf_vectorizer.pkl not found!")
        
        # Load models
        print("Loading models...")
        model = joblib.load('knn_model.pkl')
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        model_loaded = True
        print("✅ Models loaded successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to load models: {str(e)}"
        )
        
class TextInput(BaseModel):
    text: str = Field(..., description="Teks percakapan yang akan diprediksi")
    
class PredictionResponse(BaseModel):
    text: str
    prediction: str
    is_fraud: bool
    fraud_probability: float
    label: int
    timestamp: str
    
def predict_single_text(text: str) -> dict:
    load_models()
    
    text_tfidf = tfidf_vectorizer.transform([text])
    
    prediction = model.predict(text_tfidf)[0]
    proba = model.predict_proba(text_tfidf)[0]
    
    is_fraud = bool(prediction == 1)
    fraud_prob = float(proba[1]) if len(proba) > 1 else 0.0
    
    return {
        "text": text,
        "prediction": "FRAUD" if is_fraud else "NORMAL",
        "is_fraud": is_fraud,
        "fraud_probability": round(fraud_prob * 100, 2),
        "label": int(prediction),
        "timestamp": datetime.now().isoformat()
    }
    
@app.get("/", tags=["Root"])
def root():
    """Root endpoint - informasi API"""
    return {
        "name": "Fraud Detection API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model_loaded,
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint"""
    try:
        load_models()
        return {
            "status": "healthy",
            "model_loaded": model_loaded,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "model_loaded": False
        }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_fraud(input: TextInput):
    """
    Prediksi fraud untuk single text
    """
    try:
        result = predict_single_text(input.text)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    try:
        load_models()
    except Exception as e:
        print(f"⚠️ Warning: Could not load models on startup: {e}")
        print("Models will be loaded on first request")
        
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🚀 Starting Fraud Detection API Server")
    print("="*70)
    print("\n📡 Server will be available at:")
    print("   - http://localhost:5000")
    print("\n📚 API Documentation:")
    print("   - http://localhost:5001/")

    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
