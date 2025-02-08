import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBClassifier

# Load model XGBoost yang sudah di-train dalam format JSON
model = XGBClassifier()
model.load_model('trained_model.json')  # Ganti dengan path model JSON yang benar

def predict_employee_attrition(input_data):
    # Salin DataFrame asli untuk mempertahankan semua kolom input
    original_data = input_data.copy()

    # Proses data untuk fitur yang digunakan dalam model
    ordinal_mappings = {
        'Work-Life Balance': {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4},
        'Job Satisfaction': {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4},
        'Performance Rating': {'Low': 1, 'Below Average': 2, 'Average': 3, 'High': 4},
        'Education Level': {'High School': 1, 'Associate Degree': 2, 'Bachelor’s Degree': 3, 'Master’s Degree': 4, 'PhD': 5},
        'Job Level': {'Entry': 1, 'Mid': 2, 'Senior': 3},
        'Company Reputation': {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4},
        'Employee Recognition': {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4},
        'Company Size' : {'Small': 1, 'Medium': 2, 'Large': 3},
        'Monthly Income': {'Low': 1, 'Medium': 2, 'High': 3}
    }

    for col, mapping in ordinal_mappings.items():
        input_data[col] = input_data[col].map(mapping)

    # Binary encoding for binary columns
    binary_columns = ['Remote Work', 'Leadership Opportunities', 'Innovation Opportunities', 'Overtime', 'Gender']
    binary_mapping = {'No': 0, 'Yes': 1, 'Male': 0, 'Female': 1}

    for col in binary_columns:
        input_data[col] = input_data[col].map(binary_mapping)

    # Creating dummy variables for Marital Status
    input_data = pd.get_dummies(input_data, columns=['Marital Status'], drop_first=True)
    
    # Ensure the input data has the same features as the model was trained on
    selected_features = [
        'Gender', 'Work-Life Balance', 'Job Satisfaction', 'Performance Rating', 'Overtime', 'Education Level',
        'Remote Work', 'Company Reputation', 'Years at Company', 'Number of Promotions', 'Distance from Home',
        'Number of Dependents', 'Marital Status_Single', 'Marital Status_Married', 'Job Level', 'Monthly Income'
    ]
    input_data = input_data[selected_features]
    scaler = StandardScaler()
    for col in input_data.columns:
        input_data[col] = scaler.fit_transform(input_data[[col]])
 
    # Make predictions
    predictions = model.predict(input_data)

    # Add predictions to the DataFrame
    input_data['Attrition Prediction'] = predictions

    # Calculate additional columns
    input_data['Annual Income'] = original_data['Monthly Income'] * 12
    input_data['Replacement Cost'] = (original_data['Monthly Income'] * 12) * 0.33

    # Decode the predicted values back to the original labels
    input_data['Attrition Prediction'] = input_data['Attrition Prediction'].map({0: 'Stayed', 1: 'Left'})

    # Combine the predicted values with the original input data (keeping unused columns)
    # The `original_data` has all the original columns (including those not used by the model)
    result_data = pd.concat([original_data, input_data[['Attrition Prediction', 'Annual Income', 'Replacement Cost']]], axis=1)

    return result_data

# Streamlit UI
st.title('Employee Attrition Prediction')
st.write('Upload a CSV file to predict employee attrition.')
st.write('Dataset for test: https://bit.ly/Dummy-Data-for-Test')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the uploaded file into a DataFrame
    input_data = pd.read_csv(uploaded_file)
    
    # Display the input data
    st.write("Input Data:")
    st.dataframe(input_data)
    
    # Add a button for predictions
    if st.button("Predict"):
        # Process the input data and make predictions
        result = predict_employee_attrition(input_data)
        
        # Display the result in the app
        st.write("Predicted Attrition and Additional Information:")
        st.dataframe(result)
        
        # Optionally, provide the result as a downloadable CSV
        result_csv = result.to_csv(index=False)
        st.download_button(label="Download Result CSV", data=result_csv, file_name="predictions.csv", mime="text/csv")
