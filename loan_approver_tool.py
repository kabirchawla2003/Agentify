import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from langchain_core.tools import tool
from typing import Dict, Any, Union
import json

REQUIRED_FIELDS = {
    'Gender': ['Male', 'Female'],
    'Married': ['Yes', 'No'],
    'Dependents': ['0', '1', '2', '3+'],
    'Education': ['Graduate', 'Not Graduate'],
    'Self_Employed': ['Yes', 'No'],
    'ApplicantIncome': float,
    'CoapplicantIncome': float,
    'LoanAmount': float,
    'Loan_Amount_Term': float,
    'Credit_History': [0, 1],
    'Property_Area': ['Rural', 'Urban', 'Semiurban']
}

def get_loan_form() -> str:
    """Returns a formatted string with the loan application form"""
    return """Please provide the following details for loan approval:
1. Gender (Male/Female)
2. Marital Status (Yes/No)
3. Number of Dependents (0/1/2/3+)
4. Education (Graduate/Not Graduate)
5. Self Employed (Yes/No)
6. Applicant Income (number)
7. Co-applicant Income (number)
8. Loan Amount (number)
9. Loan Term (in months)
10. Credit History (0 for poor, 1 for good)
11. Property Area (Rural/Urban/Semiurban)

Example format: 
Gender:Male, Married:Yes, Dependents:2, Education:Graduate, Self_Employed:No, ApplicantIncome:5000, CoapplicantIncome:2000, LoanAmount:200, Loan_Amount_Term:360, Credit_History:1, Property_Area:Urban"""

def parse_input(input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Parse input data from either string or dictionary format"""
    if isinstance(input_data, dict):
        # Convert all keys to proper format
        return {k.replace(' ', '_'): v for k, v in input_data.items()}
        
    # Parse string input in format "key:value,key:value"
    try:
        # Split by commas and then by colons
        pairs = [pair.strip() for pair in input_data.split(',')]
        result = {}
        for pair in pairs:
            if ':' in pair:
                key, value = pair.split(':', 1)
                key = key.strip().replace(' ', '_')  # Replace spaces with underscores
                value = value.strip()
                
                # Convert numeric values
                if key in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']:
                    try:
                        value = float(value)
                    except ValueError:
                        raise ValueError(f"Invalid numeric value for {key}: {value}")
                elif key == 'Credit_History':
                    try:
                        value = int(float(value))  # Handle both int and float inputs
                    except ValueError:
                        raise ValueError(f"Invalid credit history value: {value}")
                    
                result[key] = value
        return result
    except Exception as e:
        raise ValueError(f"Error parsing input: {str(e)}")

def validate_input(data: Dict[str, Any]) -> tuple[bool, str]:
    """Validate the input data against required fields and their allowed values"""
    missing_fields = []
    invalid_values = []
    
    # Normalize data keys
    data = {k.replace(' ', '_'): v for k, v in data.items()}
    
    for field, allowed_values in REQUIRED_FIELDS.items():
        normalized_field = field.replace(' ', '_')
        if normalized_field not in data:
            missing_fields.append(field)
            continue
            
        value = data[normalized_field]
        if allowed_values == float:
            try:
                data[normalized_field] = float(value)  # Convert to float and store back
            except (ValueError, TypeError):
                invalid_values.append(f"{field} must be a number")
        elif isinstance(value, str):
            # Case-insensitive comparison for string values
            if not any(str(value).lower() == str(av).lower() for av in allowed_values):
                invalid_values.append(f"{field} must be one of {allowed_values}")
        elif isinstance(value, (int, float)) and allowed_values != float:
            if not any(value == float(av) if isinstance(av, str) else value == av for av in allowed_values):
                invalid_values.append(f"{field} must be one of {allowed_values}")
    
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    if invalid_values:
        return False, f"Invalid values: {'; '.join(invalid_values)}"
    
    return True, ""

@tool
def predict_loan_approval(input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Predict loan approval based on applicant data.
    
    Args:
        input_data: Dictionary or string containing applicant details with the following required fields:
            - Gender (Male/Female)
            - Married (Yes/No)
            - Dependents (0/1/2/3+)
            - Education (Graduate/Not Graduate)
            - Self_Employed (Yes/No)
            - ApplicantIncome (number)
            - CoapplicantIncome (number)
            - LoanAmount (number)
            - Loan_Amount_Term (number)
            - Credit_History (0/1)
            - Property_Area (Rural/Urban/Semiurban)
    
    Returns:
        Dictionary with prediction result and confidence score
    """
    try:
        # If input is just a request for the form
        if isinstance(input_data, str) and input_data.lower() in ['form', 'help', 'details', 'requirements']:
            return {"form": get_loan_form()}
            
        # Parse input if it's a string
        data = parse_input(input_data)
        
        # Validate input
        is_valid, error_message = validate_input(data)
        if not is_valid:
            return {"error": error_message}
        
        # Convert input to DataFrame
        new_data = pd.DataFrame([data])
        
        # Load the model and label encoders
        model = joblib.load("random_forest_model.pkl")
        label_encoders = joblib.load("label_encoders.pkl")
        
        # Encode categorical variables using saved encoders
        categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
        for col in categorical_cols:
            try:
                new_data[col] = label_encoders[col].transform(new_data[col])
            except ValueError as e:
                return {"error": f"Invalid value for {col}: {new_data[col].iloc[0]}. Must be one of {label_encoders[col].classes_}"}
        
        # Make prediction
        prediction = model.predict(new_data)
        prediction_proba = model.predict_proba(new_data)
        confidence = max(prediction_proba[0]) * 100
        
        result = "Approved" if prediction[0] == 1 else "Not Approved"
        
        return {
            "prediction": result,
            "confidence": f"{confidence:.2f}%",
            "factors": {
                "credit_score": data.get('Credit_History', 'N/A'),
                "income_to_loan_ratio": f"{float(data['ApplicantIncome']) / float(data['LoanAmount']):.2f}",
                "has_coapplicant": data['CoapplicantIncome'] > 0
            }
        }
        
    except Exception as e:
        return {"error": f"Error processing loan application: {str(e)}"}