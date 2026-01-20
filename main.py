import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap  # The Explainable AI Library
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from Dataset.data_preprocessing import load_and_preprocess

# IMPORTING MODELS 
# Ensure your 'models' folder has __init__.py and these files exist
from models.KNN import ChurnKNN
from models.SVM import ChurnSVM
from models.XGBoost import ChurnXGB


#MAIN EXECUTION LOOP
if __name__ == "__main__":
    # Update this path to match your actual file location
    csv_path = 'C:\\Academics\\Sem 4\\Machine learning\\ChurnPred\\Dataset\\Telco Customer Churn\\WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess(csv_path)
    
    # Initialize Base Models
    models = {
        "KNN": ChurnKNN(n_neighbors=11),
        "SVM": ChurnSVM(),
        "XGBoost": ChurnXGB()
    }
    
    # --- DEFINE HYPERPARAMETER GRIDS ---
    
    # KNN: Try different neighbors and weights
    knn_params = {
        'n_neighbors': [5, 9, 11],
        'weights': ['uniform', 'distance']
    }

    # SVM: Try different 'C' (Margin hardness)
    # Note: Reduced grid size for speed. Add more params if you have a fast PC.
    svm_params = {
        'C': [1, 10],
        'kernel': ['rbf'] 
    }

    # XGBoost: The most important one to tune
    xgb_params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.2]
    }
    
    # Dictionary mapping model names to (Model Object, Parameter Grid)
    models_to_tune = {
        "KNN": (models['KNN'].model, knn_params),
        "SVM": (models['SVM'].model, svm_params),
        "XGBoost": (models['XGBoost'].model, xgb_params)
    }

    # Variable to store the best XGBoost model for SHAP later
    best_xgb_for_shap = None

    print("\n" + "="*40)
    print("   STARTING HYPERPARAMETER TUNING")
    print("="*40)

    for name, (model_obj, params) in models_to_tune.items():
        print(f"\nTuning {name}...")
        
        # Grid Search with 3-Fold Cross Validation
        # scoring='f1' prioritizes balancing Precision and Recall
        grid = GridSearchCV(model_obj, params, cv=3, scoring='f1', n_jobs=-1, verbose=1)
        
        # Train on the SMOTE Balanced Data
        grid.fit(X_train, y_train)
        
        print(f"Best Params for {name}: {grid.best_params_}")
        
        # Predict using the best found model
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        
        print(classification_report(y_test, y_pred))

        print(f"Plotting Confusion Matrix for {name}...")
        # This automatically plots the matrix using the trained model and test data
        disp = ConfusionMatrixDisplay.from_estimator(
            best_model, 
            X_test, 
            y_test, 
            cmap=plt.cm.Blues,
            display_labels=["Stay", "Churn"]
        )
        plt.title(f"Confusion Matrix: {name}")
        plt.grid(False) # Turn off grid lines for cleaner look
        plt.show()
        
        # Capture XGBoost model for SHAP analysis
        if name == "XGBoost":
            best_xgb_for_shap = best_model

    # 3. SHAP ANALYSIS (Explainable AI) - Runs ONLY for XGBoost
    if best_xgb_for_shap is not None:
        print("\n" + "="*40)
        print("   GENERATING SHAP EXPLANATION (XGBoost)")
        print("="*40)
        
        # Access the inner Scikit-Learn wrapper of XGBoost
        # Note: If this fails, try best_xgb_for_shap directly, but get_booster() is safer for plots
        xgboost_inner = best_xgb_for_shap
        
        # Create the Explainer
        # TreeExplainer is significantly faster for XGBoost than KernelExplainer
        explainer = shap.TreeExplainer(xgboost_inner)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_test)
        
        # PLOT 1: Summary Plot (Bar Chart - Feature Importance)
        print("Plotting SHAP Summary (Bar)...")
        plt.figure()
        # 
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.title("What drives Customer Churn? (Global Importance)")
        plt.tight_layout()
        plt.show()

        # PLOT 2: Summary Plot (Dot Plot - Direction of Impact)
        print("Plotting SHAP Summary (Dot)...")
        plt.figure()
        # 
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title("Feature Impact (Red = High Value, Blue = Low Value)")
        plt.tight_layout()
        plt.show()
    else:
        print("XGBoost model was not found/trained, skipping SHAP analysis.")

    import pickle
    
    if best_xgb_for_shap is not None:
        print("\nSaving best XGBoost model to 'churn_model.pkl'...")
        with open('xgboost.pkl', 'wb') as f:
            pickle.dump(best_xgb_for_shap, f)
        print("Model saved successfully!")
    else:
        print("Error: No model to save.")

    if best_xgb_for_shap is not None:
        print("\n--- Saving Preprocessing Artifacts ---")
        
        # 1. Save the Feature Names (Crucial for aligning columns in the web app)
        # 'feature_names' is already returned by your load_and_preprocess function
        with open('model_columns.pkl', 'wb') as f:
            pickle.dump(feature_names, f)
            
        # 2. Save the Scaler
        # Since the original 'scaler' was local to the function, we create a new one
        # and fit it on the EXACT data used for training to replicate the logic.
        # Note: In a production pipeline, you would usually return the scaler from the function,
        # but fitting a fresh one on the processed training data works perfectly here.
        scaler_for_deployment = MinMaxScaler()
        scaler_for_deployment.fit(X_train) # X_train is already encoded/scaled, but this saves the range (0-1) logic
        
        with open('model_scaler.pkl', 'wb') as f:
            pickle.dump(scaler_for_deployment, f)
            
        print("✅ Scaler and Feature Names saved successfully!")
        print("   - model_columns.pkl")
        print("   - model_scaler.pkl")