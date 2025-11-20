import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

# 加载数据集
dataset = pd.read_csv("job_applicant_dataset.csv")

def calculate_group_success_rate(data, group_column):
    """计算指定分组的匹配成功率"""
    grouped = data.groupby(group_column)["Best Match"].mean().reset_index()
    grouped.columns = [group_column, "Success Rate"]
    return grouped

# 1. Load Pre-trained Model
st.title("🤖 BERT + XGBoost Resume Matching System")
st.write("Input the information and get prediction")

model = joblib.load("xgb_resume_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
bert_model = SentenceTransformer('all-MiniLM-L6-v2',device ="cpu")

# 2. User Input
age = st.number_input("Age", min_value=18, max_value=65, value=30)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
race = st.selectbox("Race", ["White/Caucasian", "Mongoloid/Asian", "Negroid/Black"])
ethnicity = st.text_input("Nationality", "Chinese")
resume = st.text_area("Resume Text", "Proficient in Python, Machine Learning, Data Analysis...")
job_desc = st.text_area("Job Description", "Responsible for building predictive models...")

# 3. Prediction
if st.button("🔮 Predict"):
    new_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Race": race,
        "Ethnicity": ethnicity,
        "Resume": resume,
        "Job Description": job_desc
    }])

    # BERT embedding
    text_embedding = bert_model.encode([resume + " " + job_desc])

    # Tabular features
    X_tabular = preprocessor.transform(new_data)

    # Concatenation
    X_final = np.hstack([text_embedding, X_tabular.toarray()])

    # Model Calling
    pred = model.predict(X_final)[0]
    proba = model.predict_proba(X_final)[0][1]

    if pred == 1:
        st.success(f"✅ Prediction Outcome: Match (Matching Probability {proba:.2f})")
    else:
        st.error(f"❌ Prediction Outcome: Not Match (Matching Probability {proba:.2f})")

    # Data Visualization
    fig, ax = plt.subplots()
    ax.bar(["Not Match", "Match"], model.predict_proba(X_final)[0], color=["red", "green"])
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

# 4. Group Analysis
st.header("Group Analysis")
tab1, tab2, tab3 = st.tabs(["Gender", "Ethnicity", "Race"])

with tab1:
    st.header("Gender Analysis")
    gender_data = calculate_group_success_rate(dataset, "Gender")
    fig, ax = plt.subplots()
    ax.bar(gender_data["Gender"], gender_data["Success Rate"], color=["blue", "pink", "gray"])
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

with tab2:
    st.header("Ethnicity Analysis")
    ethnicity_data = calculate_group_success_rate(dataset, "Ethnicity")
    fig, ax = plt.subplots()
    ax.bar(ethnicity_data["Ethnicity"], ethnicity_data["Success Rate"], color="green")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

with tab3:
    st.header("Race Analysis")
    race_data = calculate_group_success_rate(dataset, "Race")
    fig, ax = plt.subplots()
    ax.bar(race_data["Race"], race_data["Success Rate"], color=["red", "yellow", "black"])
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1)
    st.pyplot(fig)
