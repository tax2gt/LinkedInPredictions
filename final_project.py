import pandas as pd
import numpy as np
# import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x

def predict_linkedin_usage(model, income, education, parent, married, female, age):
    input_features = pd.DataFrame(
        [[income, education, parent, married, female, age]],
        columns=['income', 'educ2', 'parent', 'married', 'female', 'age']
    )
    probability = model.predict_proba(input_features)[0][1]
    return probability

    
ss = pd.DataFrame({
    "sm_li":s["web1h"].apply(clean_sm),
    "income":np.where(s["income"] > 9, np.nan, s["income"]),
    "education":np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent":s["par"].apply(clean_sm),
    "married":s["marital"].apply(clean_sm),
    "female":np.where(s["gender"] == 2,1,0),
    "age":np.where(s["age"] > 98, np.nan, s["age"])
})

ss = ss.dropna()

y = ss["sm_li"]
X = ss[["income","education","parent","married","female","age"]]

X_train, X_test, y_train, y_test = train_test_split(X.values,
                                                    y,
                                                    stratify=y,       
                                                    test_size=0.2,    
                                                    random_state=987) 

lr = LogisticRegression(class_weight="balanced")
lr.fit(X_train, y_train)

# income = st.number_input("Income (1-9):", min_value=1, max_value=9, value=5)
# education = st.number_input("Education (1-8):", min_value=1, max_value=8, value=4)
# parent = st.radio("Are you a parent?", ("Yes", "No"))
# married = st.radio("Are you married?", ("Yes", "No"))
# female = st.radio("Are you female?", ("Yes", "No"))
# age = st.number_input("Age:", min_value=18, max_value=100, value=30)

# parent = 1 if parent == "Yes" else 0
# married = 1 if married == "Yes" else 0
# female = 1 if female == "Yes" else 0

# new_data = np.array([[income, education, parent, married, female, age]])
# prediction = lr.predict(new_data)
# probability = lr.predict_proba(new_data)[0][1]

# if prediction == 1:
#     st.write("This person is predicted to be a LinkedIn user.")
# else:
#     st.write("This person is predicted to be a non-LinkedIn user.")

# st.write(f"Probability of using LinkedIn: {probability:.2f}")

prob_42 = predict_linkedin_usage(lr, income=8, education=7, parent=0, married=1, female=1, age=42)
print(f"Probability of LinkedIn usage (42 years old): {prob_42:.2f}")

#hello