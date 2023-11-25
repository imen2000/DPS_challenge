from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model_data = joblib.load('saved_steps.joblib')


model = model_data["model"]
X= model_data["training_data"]


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Extracting values from JSON data
    category = data['Category']
    accident_type = data['Accident-type']
    year = data['Year']
    month = data['Month']
    
    # Creating a DataFrame from the received data
    input_data = pd.DataFrame({
        'Category': [category],
        'Accident-type': [accident_type],
        'Year': [year],
        'Month': [month]
    })
    
    # Perform one-hot encoding for Category and Accident-type columns
    input_data = pd.get_dummies(input_data, drop_first=True, columns=['Category', 'Accident-type'])
    
    # Fill missing columns with default values (0)
    missing_cols = set(X.columns) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0
    
    # Ensure columns are in the same order as during model training
    input_data = input_data[X.columns]
    
    # Make predictions using the loaded model
    predicted_value = model.predict(input_data)
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': predicted_value[0]})

if __name__ == '__main__':
    app.run(debug=True)
