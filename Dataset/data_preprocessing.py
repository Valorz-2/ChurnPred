import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter

# 1. Load Data
df = pd.read_csv('C:\\Academics\\Sem 4\\Machine learning\\ChurnPred\\Dataset\\Telco Customer Churn\\WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 2. Fix 'TotalCharges'
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(0, inplace=True)

# 3. Drop useless columns
df.drop(columns=['customerID'], inplace=True)

# 4. Encoding
# Label Encoding
le = LabelEncoder()
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# One-Hot Encoding
df = pd.get_dummies(df, columns=['MultipleLines', 'InternetService', 'OnlineSecurity',
                                 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                 'StreamingTV', 'StreamingMovies', 'Contract', 
                                 'PaymentMethod'], drop_first=True)

# 5. Scaling
scaler = MinMaxScaler()
X = df.drop('Churn', axis=1)
y = df['Churn']

X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 6. Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Class Balancing using SMOTE
print(f"Original Training Class Distribution: {Counter(y_train)}")

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Fit SMOTE on Training Data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Balanced Training Class Distribution: {Counter(y_train_resampled)}")

# 7. Convert to NumPy arrays as it is faster in mathematical calculations like in KNN and SVM
X_train_np = X_train_resampled.values
y_train_np = y_train_resampled.values

X_test_np = X_test.values
y_test_np = y_test.values

print("Data Preprocessing & SMOTE Complete.")
print(f"Final Training Shape: {X_train_np.shape}")
print(f"Final Test Shape: {X_test_np.shape}")