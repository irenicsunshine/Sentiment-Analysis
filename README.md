# Sentiment Analysis Project

A complete sentiment analysis system built from scratch, capable of classifying text as positive or negative sentiment.

## Project Structure
- `data/`: Contains datasets
- `src/`: Source code for preprocessing, modeling, and evaluation
- `models/`: Saved trained models
- `notebooks/`: Exploratory data analysis and experiments
- `results/`: Evaluation results and visualizations

## Setup
1. Install dependencies:
```
pip install -r requirements.txt
```

2. Download required NLTK resources:
```
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Usage
- Train the model: `python src/train_model.py`
- Analyze text: `python src/analyze_text.py "Your text here"`
- Run evaluation: `python src/evaluate_model.py`
