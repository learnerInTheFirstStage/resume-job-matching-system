import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer
import joblib

# 1. 加载数据
df = pd.read_csv("job_applicant_dataset.csv")

# 特征 & 标签
X = df[["Age", "Gender", "Race", "Ethnicity", "Resume", "Job Description"]]
y = df["Best Match"]

# 2. 文本向量化 (BERT)
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_text_embeddings(resumes, jobs):
    texts = (resumes + " " + jobs).tolist()
    embeddings = bert_model.encode(texts, show_progress_bar=True)
    return embeddings

X_text = get_text_embeddings(X["Resume"], X["Job Description"])

# 3. 处理非文本特征
categorical = ["Gender", "Race", "Ethnicity"]
numeric = ["Age"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", StandardScaler(), numeric)
    ],
    remainder="drop"
)

X_tabular = preprocessor.fit_transform(X)

# 拼接 BERT 向量 + 其他特征
X_final = np.hstack([X_text, X_tabular.toarray()])

# 4. 划分数据
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# 5. 训练 XGBoost
model = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train, y_train)

# 6. 评估
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

# 7. 保存模型和预处理器
joblib.dump(model, "xgb_resume_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
print("✅ 模型已保存")
