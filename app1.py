import streamlit as st
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from streamlit_shap import st_shap

# Load the dataset
customer = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Drop customerID as it's not useful for prediction
customer.drop(columns=['customerID'], inplace=True)

# Convert TotalCharges to numeric (handling non-numeric values)
customer["TotalCharges"] = pd.to_numeric(customer["TotalCharges"], errors='coerce')

# Fill missing values
customer.fillna(customer.median(numeric_only=True), inplace=True)

# Encode categorical variables
label_encoders = {}
for col in customer.select_dtypes(include=['object']).columns:
    if col != "Churn":  # Exclude target variable
        le = LabelEncoder()
        customer[col] = le.fit_transform(customer[col])
        label_encoders[col] = le

# Encode target variable
customer["Churn"] = customer["Churn"].map({"Yes": 1, "No": 0})

# Define features and target
X = customer.drop("Churn", axis=1)
y = customer["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Model training
clf = RandomForestClassifier(random_state=1)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# SHAP explainer
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# Streamlit app
st.title("SHAP Analysis for Telco Customer Churn")

# Part 1: General SHAP Analysis
st.header("Part 1: General SHAP Analysis")

# Display classification report
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# Summary plot for class 1
st.subheader("Summary Plot for Class 1")
fig, ax = plt.subplots()
if isinstance(shap_values, list):  
    shap.summary_plot(shap_values[1], X_test, show=False)  # Correct class index
else:
    shap.summary_plot(shap_values, X_test, show=False)
st.pyplot(fig)

# Part 2: Individual Input Prediction & Explanation
st.header("Part 2: Individual Input Prediction & Explanation")

# Input fields for features
input_data = {}
for feature in X.columns:
    if feature in label_encoders:  # Handle categorical values properly
        options = list(label_encoders[feature].classes_)
        input_data[feature] = st.selectbox(f"Select {feature}:", options)
        input_data[feature] = label_encoders[feature].transform([input_data[feature]])[0]  
    else:
        input_data[feature] = st.number_input(f"Enter {feature}:", value=float(X_test[feature].mean()))

# Create a DataFrame from input data
input_df = pd.DataFrame([input_data])

# Make prediction
prediction = clf.predict(input_df)[0]
probability = clf.predict_proba(input_df)[0][1]

# Display prediction
st.write(f"**Prediction:** {'Churn' if prediction == 1 else 'No Churn'}")
st.write(f"**Churn Probability:** {probability:.2f}")

# SHAP explanation for the input
shap_values_input = explainer.shap_values(input_df)

# Force plot (FIXED: Convert to compatible format)
st.subheader("Force Plot for Class 1")
if isinstance(shap_values_input, list):  
    force_plot = shap.force_plot(
        explainer.expected_value[1], shap_values_input[1], input_df
    )
else:
    force_plot = shap.force_plot(
        explainer.expected_value, shap_values_input, input_df
    )
st_shap(force_plot, height=200, width=1000)

# Decision plot (Ensure correct format)
st.subheader("Decision Plot for Class 1")
if isinstance(shap_values_input, list):
    st_shap(shap.decision_plot(explainer.expected_value[1], shap_values_input[1], input_df.columns))
else:
    st_shap(shap.decision_plot(explainer.expected_value, shap_values_input, input_df.columns))
