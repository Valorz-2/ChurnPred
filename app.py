from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# ==========================================
# 1. LOAD SAVED ARTIFACTS
# ==========================================
# Load the trained XGBoost model
model = pickle.load(open('xgboost.pkl', 'rb'))

# Load the Scaler (for Tenure, MonthlyCharges, TotalCharges)
scaler = pickle.load(open('model_scaler.pkl', 'rb'))

# Load the Column Names (to align One-Hot Encoding)
model_columns = pickle.load(open('model_columns.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # ==========================================
    # 2. GET DATA FROM FORM
    # ==========================================
    # We collect all inputs into a dictionary
    input_data = {
        'gender': request.form['gender'],
        'SeniorCitizen': int(request.form['SeniorCitizen']),
        'Partner': request.form['Partner'],
        'Dependents': request.form['Dependents'],
        'tenure': float(request.form['tenure']),
        'PhoneService': request.form['PhoneService'],
        'MultipleLines': request.form['MultipleLines'],
        'InternetService': request.form['InternetService'],
        'OnlineSecurity': request.form['OnlineSecurity'],
        'OnlineBackup': request.form['OnlineBackup'],
        'DeviceProtection': request.form['DeviceProtection'],
        'TechSupport': request.form['TechSupport'],
        'StreamingTV': request.form['StreamingTV'],
        'StreamingMovies': request.form['StreamingMovies'],
        'Contract': request.form['Contract'],
        'PaperlessBilling': request.form['PaperlessBilling'],
        'PaymentMethod': request.form['PaymentMethod'],
        'MonthlyCharges': float(request.form['MonthlyCharges']),
        'TotalCharges': float(request.form['TotalCharges'])
    }

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # ==========================================
    # 3. PREPROCESSING (REPLICATING MAIN.PY)
    # ==========================================
    
    # A. Label Encoding (Manual Mapping for Binary Columns)
    # Your main.py used LabelEncoder, which typically maps:
    # No -> 0, Yes -> 1 | Female -> 0, Male -> 1
    binary_mapping = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0, 'No internet service': 0, 'No phone service': 0}
    
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        df[col] = df[col].map(binary_mapping)

    # B. One-Hot Encoding
    # This converts 'InternetService' -> 'InternetService_Fiber optic', etc.
    df = pd.get_dummies(df)

    # C. ALIGN COLUMNS (CRITICAL STEP)
    # The single input dataframe won't have all columns (e.g., if user selects DSL, 
    # there won't be a 'Fiber optic' column). We reindex to match the training data.
    # missing columns are filled with 0.
    df = df.reindex(columns=model_columns, fill_value=0)

    # D. Scaling
    # Apply the loaded scaler to normalize tenure and charges
    df_scaled = scaler.transform(df)

    # ==========================================
    # 4. PREDICTION
    # ==========================================
    # Get the probability of CHURN (Class 1)
    # probability variable usually returns [[prob_0, prob_1]]
    churn_risk_score = model.predict_proba(df_scaled)[0][1]

    # Define a stricter threshold (e.g., 0.40 instead of default 0.50)
    # This means if risk is > 40%, we flag it!
    threshold = 0.40

    if churn_risk_score > threshold:
        result_text = "High Risk (Likely to Churn) ⚠️"
        result_color = "#dc3545" # Red
        # Show confidence of CHURN, not STAY
        display_confidence = round(churn_risk_score * 100, 2)
    else:
        result_text = "Low Risk (Likely to Stay) ✅"
        result_color = "#28a745" # Green
        # Show confidence of STAY
        display_confidence = round((1 - churn_risk_score) * 100, 2)

    return render_template('index.html', 
                           prediction_text=result_text,
                           confidence_text=f"Confidence: {display_confidence}%",
                           result_color=result_color)

if __name__ == "__main__":
    app.run(debug=True)