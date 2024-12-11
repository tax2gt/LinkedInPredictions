import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Load dataset
s = pd.read_csv("social_media_usage.csv")

# Clean Social Media function
def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x

# Function to predict LinkedIn usage probability
def predict_linkedin_usage(model, income, education, parent, married, female, age):
    input_features = pd.DataFrame(
        [[income, education, parent, married, female, age]],
        columns=['income', 'educ2', 'parent', 'married', 'female', 'age']
    )
    probability = model.predict_proba(input_features)[0][1]
    return probability

# Income and Education Labels
income_labels = {
    1: "Less than $10,000",
    2: "$10,000 to under $20,000",
    3: "$20,000 to under $30,000",
    4: "$30,000 to under $40,000",
    5: "$40,000 to under $50,000",
    6: "$50,000 to under $75,000",
    7: "$75,000 to under $100,000",
    8: "$100,000 to under $150,000",
    9: "$150,000 or more"
}

education_labels = {
    1: "Less than high school",
    2: "High school incomplete",
    3: "High school graduate",
    4: "Some college, no degree",
    5: "Two-year associate degree from a college or university",
    6: "Four-year college or university degree",
    7: "Some postgraduate or professional schooling, no postgraduate degree",
    8: "Postgraduate or professional degree"
}

# Preparing data
ss = pd.DataFrame({
    "sm_li": s["web1h"].apply(clean_sm),
    "income": np.where(s["income"] > 9, np.nan, s["income"]),
    "education": np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent": s["par"].apply(clean_sm),
    "married": s["marital"].apply(clean_sm),
    "female": np.where(s["gender"] == 2, 1, 0),
    "age": np.where(s["age"] > 98, np.nan, s["age"])
})

# Drop missing values
ss = ss.dropna()

# Target variable vectors (y) and features (X)
y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X.values, y, stratify=y, test_size=0.2, random_state=987)

# Train Logistic Regression
lr = LogisticRegression(class_weight="balanced")
lr.fit(X_train, y_train)

# Streamlit
st.title("LinkedIn Usage Prediction")

st.markdown("Please complete the fields below to predict an individual's LinkedIn usage")

income = st.selectbox("Select the individual's income range", options=[
    1, 2, 3, 4, 5, 6, 7, 8, 9], 
    format_func=lambda x: income_labels[x]  
)

education = st.selectbox("Select the individual's education level", options=[
    1, 2, 3, 4, 5, 6, 7, 8], 
    format_func=lambda x: education_labels[x]  
)

parent = st.radio("Is the individual a parent?", ("Yes", "No"))
married = st.radio("Is the individual married?", ("Yes", "No"))
female = st.radio("Is the individual female?", ("Yes", "No"))
age = st.number_input("Age:", min_value=0, max_value=100, value=18)

parent = 1 if parent == "Yes" else 0
married = 1 if married == "Yes" else 0
female = 1 if female == "Yes" else 0

new_data = np.array([[income, education, parent, married, female, age]])

# Make prediction
prediction = lr.predict(new_data)
probability = lr.predict_proba(new_data)[0][1]

if prediction == 1:
    title_text = "This individual is predicted to be a LinkedIn user."
else:
    title_text = "This individual is predicted to be a LinkedIn non-user."

# Visualization
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=probability,
    title={'text': title_text},
    gauge={
        'axis': {'range': [0, 1]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 0.5], 'color': "red"},
            {'range': [0.5, 1], 'color': "green"}
        ]
    }
))

st.plotly_chart(fig)
