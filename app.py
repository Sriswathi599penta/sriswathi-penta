from flask import Flask, render_template, request, jsonify
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

model = joblib.load('Xgboost.joblib')  
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

analyzer = SentimentIntensityAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    

    predicted_probability = analyzer.polarity_scores(text)['compound']
    xgb_prediction = "Positive" if predicted_probability > 0 else "Negative" if predicted_probability < 0 else "Neutral"

    result = {
        'vader_sentiment': xgb_prediction,
        'probability_score': predicted_probability
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
