from flask import Flask, render_template, request , redirect
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load model bundle
model_bundle = joblib.load('model.joblib')
model = model_bundle['model']
columns = model_bundle['columns']
scaler = model_bundle['scaler']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return redirect('/')
    elif request.method == 'POST':
        # Get input values from the form
        gender = request.form['gender']
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = request.form['ever_married']
        work_type = request.form['work_type']
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking_status = request.form['smoking_status']

        # Create DataFrame from form input
        input_dict = {
            'age': age,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'gender': gender,
            'ever_married': ever_married,
            'work_type': work_type,
            'smoking_status': smoking_status
        }

        input_df = pd.DataFrame([input_dict])

        # One-hot encoding
        input_df = pd.get_dummies(input_df, drop_first=False)

        # Reindex to match training columns
        input_df = input_df.reindex(columns=columns, fill_value=0)

        # Normalize numeric features
        input_df[['age', 'bmi', 'avg_glucose_level']] = scaler.transform(input_df[['age', 'bmi', 'avg_glucose_level']])

        # Predict
        pred_prob = model.predict_proba(input_df)[0][1]
        prediction = int(pred_prob >= 0.5)

        result = 'Yes' if prediction == 1 else 'No'

        return render_template('result.html', prediction=result, probability=f"{pred_prob:.2%}")

if __name__ == '__main__':
    app.run(debug=True)