# 🤖 Resume Screening System (BERT + XGBoost)

An intelligent resume screening system powered by **Sentence-BERT** embeddings and **XGBoost** classification.  
This project includes both model training (`train.py`) and an interactive web interface (`app2.py`) built with **Streamlit**.

---

## 📘 Overview

This system aims to automatically assess how well a candidate’s resume matches a given job description.  
It uses:
- **BERT embeddings** to capture semantic meaning from text data (resume + job description)
- **XGBoost** to classify whether a candidate is a *“Best Match”*
- **Streamlit** for an easy-to-use web interface

---

## 🧩 Project Components

| File | Description |
|------|--------------|
| `train.py` | Script to train the XGBoost model using BERT text embeddings and tabular features. Saves model & preprocessor. |
| `app2.py` | Streamlit web app for prediction using the saved model. |
| `job_applicant_dataset.csv` | (Required for training) Dataset containing candidate info and labels. |
| `xgb_resume_model.pkl` | Trained model file (auto-generated after training). |
| `preprocessor.pkl` | Data preprocessing pipeline (auto-generated after training). |

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <repo>
cd <your-repo-name>
```

### 2. Create a Local Conda Environment
```bash
conda create -p ./env python=3.10
conda activate ./env
```

### 3. Install Dependencies
```bash
conda install pandas numpy matplotlib joblib scikit-learn
pip install streamlit sentence-transformers xgboost
```