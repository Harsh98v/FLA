from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load('final_model.sav')

@app.route('/predict_salary', methods=['POST'])
def predict_salary():
    try:
        # Get years of experience from request data
        years_experience = float(request.json['yearsExperience'])
        # Predict salary
        predicted_salary = model.predict([[years_experience]])[0][0]
        return jsonify(predicted_salary)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
