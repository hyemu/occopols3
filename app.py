from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

def load_and_preprocess_data():
    # Load data
    df = pd.read_excel("Student-Employability-Datasets.xlsx")

    # Map the target variable (CLASS) to binary values
    df['CLASS'] = df['CLASS'].map({'Employable': 1, 'LessEmployable': 0})
   
    # Features and target
    X = df.loc[:, "GENERAL APPEARANCE":"Student Performance Rating"]
    y = df["CLASS"]
   
    print("\nClass distribution:")
    print(y.value_counts(normalize=True))
   
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)  # Resample the dataset
   
    return X_resampled, y_resampled

def train_model(X, y):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Use GridSearch to find the best k for KNN
    param_grid = {'n_neighbors': np.arange(1, 21)}  # Try k values from 1 to 20
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid_search.fit(X_train_scaled, y_train)
    
    best_k = grid_search.best_params_['n_neighbors']
    print(f"Best k found: {best_k}")
    
    # Train KNN with the best k
    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train_scaled, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel accuracy: {accuracy:.2f}")
    
    return model, scaler

def apply_threshold(input_data):
    # Print the received input data for debugging
    print(f"Received input data for thresholding: {input_data}")

    # Compute the average score across input features
    avg_score = np.mean(input_data)
    print(f"Average score: {avg_score:.2f}")
    
    # Dynamically adjust the threshold (this could be improved)
    threshold = 3.5
    return "Employable" if avg_score >= threshold else "Less Employable"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print('Received data:', data)

    # Define the expected features (these should match the column names in X)
    expected_columns = [
            "general_appearance", "manner_of_speaking", "physical_condition", 
            "mental_alertness", "self_confidence", "ability_to_present_ideas", 
            "communication_skills", "student_performance_rating"
        ]


    # Prepare raw input data for prediction (using 3 as a fallback if a value is missing)
    input_data = []
    for attr in expected_columns:
        input_value = data.get(attr, 3)  # Default to 3 if the attribute is missing
        input_data.append(input_value)

    # Print the received input data for thresholding
    print(f"Received input data for thresholding: {input_data}")

    # Apply the thresholding logic using raw input values (no scaling for thresholding)
    threshold_result = apply_threshold(input_data)

    # Scale input data for prediction (only for the model)
    input_scaled = scaler.transform([input_data])

    # Make KNN prediction and calculate probabilities
    knn_prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    # Combine KNN and threshold results for final decision
    final_result = "Employable" if (knn_prediction == 1 or threshold_result == "Employable") else "Less Employable"

    # Get the indices of the nearest neighbors
    distances, indices = model.kneighbors(input_scaled)

    # Get the class of each neighbor
    neighbor_classes = y.iloc[indices[0]].values
    neighbor_classes = ["Employable" if nc == 1 else "Less Employable" for nc in neighbor_classes]

    print('KNN Prediction:', "Employable" if knn_prediction == 1 else "Less Employable")
    print('Threshold Result:', threshold_result)
    print('Final Prediction:', final_result)
    print('Probabilities:', probabilities)
    print('Neighbor classes:', neighbor_classes)

    # Return response
    return jsonify({
        "prediction": final_result,
        "knn_prediction": "Employable" if knn_prediction == 1 else "Less Employable",
        "threshold_result": threshold_result,
        "probabilities": probabilities.tolist(),
        "neighbor_classes": neighbor_classes
    })


if __name__ == '__main__':
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    model, scaler = train_model(X, y)

    # Run Flask app
    app.run(debug=True)
