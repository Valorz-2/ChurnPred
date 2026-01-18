from sklearn.svm import SVC

class ChurnSVM:
    def __init__(self, kernel='linear', C=1.0):
        # We need probability=True if we ever want to do advanced analysis later, 
        # but for standard predictions, default is fine.
        self.model = SVC(kernel=kernel, C=C, random_state=42)
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)