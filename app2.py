import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

# 1. 加载模型
st.title("🤖 BERT + XGBoost 简历筛选系统")
st.write("输入候选人信息，预测是否为 **最佳匹配**")

model = joblib.load("xgb_resume_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
bert_model = SentenceTransformer('all-MiniLM-L6-v2',device ="cpu")

# 2. 用户输入
age = st.number_input("年龄", min_value=18, max_value=65, value=30)
gender = st.selectbox("性别", ["Male", "Female", "Other"])
race = st.selectbox("种族", ["White/Caucasian", "Mongoloid/Asian", "Negroid/Black"])
ethnicity = st.text_input("民族", "Chinese")
resume = st.text_area("简历文本", "Proficient in Python, Machine Learning, Data Analysis...")
job_desc = st.text_area("职位描述", "Responsible for building predictive models...")

# 3. 预测
if st.button("🔮 预测结果"):
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

    # 拼接
    X_final = np.hstack([text_embedding, X_tabular.toarray()])

    # 预测
    pred = model.predict(X_final)[0]
    proba = model.predict_proba(X_final)[0][1]

    if pred == 1:
        st.success(f"✅ 预测结果: 适合该职位 (匹配概率 {proba:.2f})")
    else:
        st.error(f"❌ 预测结果: 不适合该职位 (匹配概率 {proba:.2f})")

    # 概率可视化
    fig, ax = plt.subplots()
    ax.bar(["不匹配", "匹配"], model.predict_proba(X_final)[0], color=["red", "green"])
    ax.set_ylabel("概率")
    ax.set_ylim(0, 1)
    st.pyplot(fig)
