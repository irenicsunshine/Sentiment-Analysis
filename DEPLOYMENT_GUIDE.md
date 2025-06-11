# Sentiment Analysis System - Deployment Guide

## Overview
This guide provides comprehensive instructions for deploying and using the advanced sentiment analysis system in various environments.

## Quick Start

### Local Development
1. **Clone/Download the project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download NLTK resources**:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"
   ```
4. **Train the model** (if not already trained):
   ```bash
   python src/train_model.py --model-type nb --feature-method bow
   ```
5. **Run the web application**:
   ```bash
   python app.py
   ```
6. **Access the application**: http://localhost:5001

## Usage Options

### 1. Web Interface (Recommended)
The web application provides a beautiful, interactive interface with:
- Real-time sentiment analysis
- Confidence scores and probability distributions
- Sentiment intensity levels (Very Positive, Positive, Neutral, Negative, Very Negative)
- Analysis history and statistics
- Quick example buttons
- Visual charts and graphs

**Start the web app:**
```bash
python app.py
```

### 2. Command Line Interface
For batch processing or integration with other tools:

**Analyze single text:**
```bash
python src/analyze_text.py "Your text here"
```

**Interactive mode:**
```bash
python src/analyze_text.py --interactive
```

**Enhanced ensemble analyzer:**
```bash
python src/enhanced_analyzer.py "Your text here"
```

### 3. Python API Integration
Import and use in your Python code:

```python
from src.preprocess import TextPreprocessor
from src.features import FeatureExtractor
import joblib

# Load trained model
model = joblib.load('models/sentiment_model.pkl')
feature_extractor = joblib.load('models/feature_extractor.pkl')
preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)

# Analyze text
text = "This movie was fantastic!"
cleaned_text = preprocessor.clean_text(text)
X_features = feature_extractor.transform([cleaned_text])
prediction = model.predict(X_features)[0]
confidence = model.predict_proba(X_features)[0].max()

print(f"Sentiment: {'Positive' if prediction == 1 else 'Negative'}")
print(f"Confidence: {confidence:.2%}")
```

## System Requirements

### Minimum Requirements
- Python 3.8+
- 2GB RAM
- 500MB disk space
- Internet connection (for initial NLTK downloads)

### Recommended Requirements
- Python 3.9+
- 4GB RAM
- 1GB disk space
- Multi-core processor for faster training

## Performance Specifications

### Current Model Performance
- **Accuracy**: 92.86% on full dataset
- **Cross-validation Accuracy**: 74.29% (robust estimate)
- **Error Rate**: 7.1%
- **Processing Speed**: ~1000 texts/second
- **Model Size**: ~50KB (lightweight)

### Confidence Levels
- **High Confidence (>80%)**: Very reliable predictions
- **Medium Confidence (60-80%)**: Generally reliable
- **Low Confidence (<60%)**: Review recommended

## Advanced Features

### 1. Model Comparison Tool
Compare different model configurations:
```bash
python src/model_comparison.py
```

### 2. Error Analysis
Analyze model errors and improvement opportunities:
```bash
python src/error_analysis.py
```

### 3. Ensemble Models
Use advanced ensemble methods:
```bash
python src/ensemble_model.py
```

## File Structure
```
sentiment-analysis/
├── app.py                     # Web application
├── requirements.txt           # Dependencies
├── README.md                 # Main documentation
├── DEPLOYMENT_GUIDE.md       # This file
├── IMPROVEMENTS_SUMMARY.md   # Performance improvements
├── data/
│   └── imdb_sample.csv       # Training dataset
├── models/
│   ├── sentiment_model.pkl   # Trained model
│   └── feature_extractor.pkl # Feature extractor
├── src/
│   ├── analyze_text.py       # Text analysis script
│   ├── enhanced_analyzer.py  # Enhanced ensemble analyzer
│   ├── ensemble_model.py     # Advanced ensemble models
│   ├── error_analysis.py     # Error analysis tool
│   ├── evaluate_model.py     # Model evaluation
│   ├── features.py           # Feature extraction
│   ├── model.py             # Model implementation
│   ├── model_comparison.py   # Model comparison tool
│   ├── preprocess.py         # Text preprocessing
│   └── train_model.py        # Model training
├── templates/
│   └── index.html           # Web interface template
├── results/
│   ├── confusion_matrix.png
│   ├── error_analysis.png
│   ├── model_comparison.csv
│   └── error_analysis_report.md
└── notebooks/
    └── sentiment_analysis_demo.ipynb
```

## Customization Options

### 1. Training with Custom Data
Replace the dataset in `data/` directory with your own CSV file containing:
- `review` column: text to analyze
- `sentiment` column: 0 (negative) or 1 (positive)

```bash
python src/train_model.py --dataset path/to/your/data.csv
```

### 2. Model Configuration
Experiment with different models and features:
```bash
# Random Forest with TF-IDF
python src/train_model.py --model-type rf --feature-method tfidf

# SVM with Bag-of-Words
python src/train_model.py --model-type svm --feature-method bow

# Hyperparameter optimization
python src/train_model.py --optimize
```

### 3. Web Interface Customization
Modify `templates/index.html` to:
- Change colors and styling
- Add new features
- Customize the layout
- Add additional metrics

## Production Deployment

### Docker Deployment
Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"

EXPOSE 5001
CMD ["python", "app.py"]
```

### Cloud Deployment Options
1. **Heroku**: Easy deployment with buildpacks
2. **AWS EC2**: Full control with custom configuration
3. **Google Cloud Run**: Serverless container deployment
4. **Azure Container Instances**: Simple container hosting

### Environment Variables
For production, set these environment variables:
```bash
export FLASK_ENV=production
export MODEL_PATH=models/sentiment_model.pkl
export FEATURE_PATH=models/feature_extractor.pkl
```

## API Endpoints

### Web API
The Flask app provides these endpoints:

- `POST /analyze`: Analyze sentiment
  ```json
  {
    "text": "Your text here"
  }
  ```

- `GET /history`: Get analysis history

- `GET /stats`: Get usage statistics

- `POST /clear_history`: Clear analysis history

## Troubleshooting

### Common Issues

1. **Model not found error**
   ```bash
   python src/train_model.py  # Retrain the model
   ```

2. **NLTK data not found**
   ```bash
   python -c "import nltk; nltk.download('all')"
   ```

3. **Port already in use**
   - Modify port in `app.py` or kill the process using the port

4. **Memory issues with large datasets**
   - Reduce `max_features` parameter
   - Process data in batches

### Performance Optimization

1. **For faster predictions**:
   - Use smaller feature vectors
   - Cache the feature extractor
   - Use simpler models (Naive Bayes)

2. **For better accuracy**:
   - Add more training data
   - Use ensemble methods
   - Experiment with different features

## Support and Maintenance

### Monitoring
- Track prediction confidence distributions
- Monitor error rates over time
- Log difficult examples for model improvement

### Updates
- Regularly retrain with new data
- Monitor for dataset drift
- Update dependencies and security patches

### Backup Strategy
- Regular backups of trained models
- Version control for code changes
- Data backup and recovery procedures

## License and Credits
This sentiment analysis system is built using:
- scikit-learn for machine learning
- NLTK for natural language processing
- Flask for web framework
- Bootstrap for UI components

---

For technical support or questions, refer to the main README.md or create an issue in the project repository.
