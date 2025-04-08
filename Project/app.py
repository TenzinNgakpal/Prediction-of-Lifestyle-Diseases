from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained lifestyle disease prediction model
model = joblib.load("lifestyle_model.pkl")

# Health recommendations and links
RECOMMENDATIONS = {
    "Diabetes": {
        "tips": [
            "Maintain a balanced, low-sugar diet.",
            "Engage in regular physical activity.",
            "Monitor blood glucose levels regularly."
        ],
        "link": "https://www.who.int/news-room/fact-sheets/detail/diabetes"
    },
    "Heart Disease": {
        "tips": [
            "Avoid smoking and excessive alcohol.",
            "Eat a heart-healthy diet rich in vegetables and whole grains.",
            "Exercise regularly, at least 150 minutes per week."
        ],
        "link": "https://www.who.int/health-topics/cardiovascular-diseases"
    },
    "Hypertension": {
        "tips": [
            "Reduce salt intake and manage stress.",
            "Maintain a healthy weight and active lifestyle.",
            "Monitor your blood pressure regularly."
        ],
        "link": "https://www.who.int/news-room/fact-sheets/detail/hypertension"
    }
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        features = [
            data['age'],
            data['bmi'],
            data['bp'],
            data['glucose'],
            data['cholesterol'],
            data['smoking'],
            data['alcohol'],
            data['physical_activity']
        ]

        input_array = np.array([features])
        prediction_array = model.predict(input_array)[0]  # Expecting 3-class output

        results = {}
        for i, disease in enumerate(["Diabetes", "Heart Disease", "Hypertension"]):
            prediction = "Positive" if prediction_array[i] == 1 else "Negative"
            results[disease] = {
                "Risk": prediction
            }
            if prediction == "Positive":
                results[disease]["Tips"] = RECOMMENDATIONS[disease]["tips"]
                results[disease]["LearnMore"] = RECOMMENDATIONS[disease]["link"]

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
