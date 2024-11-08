import json
import numpy as np
from flask import Flask, request, jsonify
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load data and preprocessing
def load_and_preprocess_data():
    df = pd.read_excel("Student-Employability-Datasets.xlsx")
    df['CLASS'] = df['CLASS'].map({'Employable': 1, 'LessEmployable': 0})
    X = df.loc[:, "GENERAL APPEARANCE":"Student Performance Rating"]
    y = df["CLASS"]
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    param_grid = {'n_neighbors': np.arange(1, 21)}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid_search.fit(X_train_scaled, y_train)
    best_k = grid_search.best_params_['n_neighbors']
    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train_scaled, y_train)
    return model, scaler

# Apply threshold for prediction
def apply_threshold(input_data):
    avg_score = np.mean(input_data)
    threshold = 3.5
    return "Employable" if avg_score >= threshold else "Less Employable"

# Load and preprocess data
X, y = load_and_preprocess_data()
model, scaler = train_model(X, y)

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json
    expected_columns = [
        "general_appearance", "manner_of_speaking", "physical_condition", 
        "mental_alertness", "self_confidence", "ability_to_present_ideas", 
        "communication_skills", "student_performance_rating"
    ]
    input_data = [data.get(attr, 3) for attr in expected_columns]
    threshold_result = apply_threshold(input_data)
    input_scaled = scaler.transform([input_data])
    knn_prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    final_result = "Employable" if (knn_prediction == 1 or threshold_result == "Employable") else "Less Employable"
    return jsonify({
        "prediction": final_result,
        "knn_prediction": "Employable" if knn_prediction == 1 else "Less Employable",
        "threshold_result": threshold_result,
        "probabilities": probabilities.tolist(),
    })

if __name__ == "__main__":
    app.run(debug=True)
