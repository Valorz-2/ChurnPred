from xgboost import XGBClassifier

class ChurnXGB:
    def __init__(self):
        # Optimized params for Churn usually involve max_depth=3 to 5
        self.model = XGBClassifier( 
            eval_metric='logloss',
            random_state=42
        )
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
        
    # Helper to return the inner model object for SHAP analysis
    def get_booster(self):
        return self.model