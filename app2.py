import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import io
import google.generativeai as genai

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ==========================
# Streamlit settings
# ==========================
st.set_page_config(
    page_title="Resume Matcher", 
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🤖"
)

# ==========================
# Custom CSS Styling
# ==========================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Force light theme - override Streamlit defaults */
    html, body, [class*="css"] {
        background-color: #f5f7fa !important;
        color: #2c3e50 !important;
    }
    
    /* Main container styling */
    .main {
        padding: 2rem 3rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%) !important;
    }
    
    /* Streamlit app container */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%) !important;
    }
    
    /* Block container */
    .block-container {
        background: transparent !important;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Report container */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%) !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
    }
    
    /* Title styling */
    h1 {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    /* Headers */
    h2, h3 {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        color: #34495e;
        margin-top: 1.5rem;
    }
    
    /* Body text */
    .stMarkdown, .stText, p, div, span {
        font-family: 'Inter', sans-serif;
        color: #2c3e50 !important;
    }
    
    /* All text elements - force light colors */
    p, div, span, label, h1, h2, h3, h4, h5, h6 {
        color: #2c3e50 !important;
    }
    
    /* Custom card containers */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border: 1px solid #e8ecf1;
    }
    
    /* Input fields styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextArea > div > div > textarea,
    input, select, textarea {
        background-color: #ffffff !important;
        color: #2c3e50 !important;
        border: 1.5px solid #e0e6ed !important;
        border-radius: 8px;
        padding: 0.75rem;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #4a90e2;
        box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stError {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stInfo {
        background-color: #e7f3ff;
        border-left: 4px solid #4a90e2;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stWarning {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        color: #2c3e50;
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif;
        color: #7f8c8d;
        font-weight: 500;
    }
    
    /* File uploader styling */
    .stFileUploader,
    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] > div {
        background: #ffffff !important;
        border: 2px dashed #cbd5e0 !important;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        color: #2c3e50 !important;
    }
    
    .stFileUploader:hover,
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea !important;
        background: #f8f9ff !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader,
    [data-testid="stExpander"] > div:first-child {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        color: #34495e !important;
        background: #ffffff !important;
        border-radius: 8px;
        padding: 0.75rem;
    }
    
    /* Expander content */
    .streamlit-expanderContent,
    [data-testid="stExpander"] > div:last-child {
        background: #ffffff !important;
        color: #2c3e50 !important;
    }
    
    /* All expander elements */
    [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid #e0e6ed !important;
        border-radius: 8px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
    }
    
    /* Divider styling */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, #e0e6ed, transparent);
        margin: 2rem 0;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Column spacing */
    [data-testid="column"] {
        padding: 0 1rem;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #667eea;
    }
    
    /* Force all Streamlit widgets to have light backgrounds */
    [data-baseweb="select"] {
        background-color: #ffffff !important;
    }
    
    /* File uploader area */
    [data-testid="stFileUploader"] {
        background-color: #ffffff !important;
    }
    
    /* Expander content */
    .streamlit-expanderContent {
        background-color: #ffffff !important;
    }
    
    /* All containers should be light */
    div[class*="element-container"] {
        background-color: transparent !important;
    }
    
    /* Tables and dataframes */
    table, .dataframe {
        background-color: #ffffff !important;
        color: #2c3e50 !important;
    }
    
    /* Bullet points styling */
    ul, ol {
        color: #2c3e50 !important;
        padding-left: 1.5rem;
    }
    
    li {
        color: #2c3e50 !important;
        margin-bottom: 0.5rem;
        line-height: 1.6;
    }
    
    /* File uploader text */
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span {
        color: #2c3e50 !important;
    }
    
    /* Override any dark theme elements */
    [data-testid="stAppViewContainer"] > div:first-child {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%) !important;
    }
    
    /* Ensure widgets have light backgrounds */
    [data-baseweb="base-input"] {
        background-color: #ffffff !important;
    }
    
    /* Select dropdown */
    [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #2c3e50 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================
# Header Section
# ==========================
st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='font-size: 2.5rem; margin-bottom: 0.5rem;'>🤖 Intelligent Resume Screening System</h1>
        <p style='font-size: 1.1rem; color: #7f8c8d; font-weight: 400;'>AI-Powered Resume Matching & Career Insights</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================
# Load Model
# ==========================
model = joblib.load("xgb_resume_with_negatives.pkl")
preprocessor = joblib.load("preprocessor_with_negatives.pkl")
bert_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# ==========================
# Load dataset to get job role mapping
# ==========================
@st.cache_data
def load_dataset():
    df = pd.read_csv("data/job_applicant_dataset.csv", encoding="Windows-1252")
    role_to_desc = df.groupby("Job Roles")["Job Description"].first().to_dict()
    return df, role_to_desc

@st.cache_data
def load_job_market_data():
    """Load job postings and salary data"""
    try:
        postings_by_sector = pd.read_csv("data/postings_by_sector.csv")
        postings_total_us = pd.read_csv("data/postings_total_us.csv")
        salary_overview = pd.read_csv("data/salary_overview_soc.csv")
        
        # Convert month column to datetime for time series analysis
        if 'month' in postings_by_sector.columns:
            postings_by_sector['month'] = pd.to_datetime(postings_by_sector['month'])
        if 'month' in postings_total_us.columns:
            postings_total_us['month'] = pd.to_datetime(postings_total_us['month'])
        
        return postings_by_sector, postings_total_us, salary_overview
    except FileNotFoundError as e:
        st.warning(f"Some job market data files not found: {e}")
        return None, None, None

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {str(e)}")
        return None

df, role_to_desc = load_dataset()
postings_by_sector, postings_total_us, salary_overview = load_job_market_data()

# ==========================
# User Inputs Section
# ==========================
st.markdown("---")

# Create a container for better visual grouping
with st.container():
    st.markdown("""
        <div style='background: white; padding: 1.5rem; border-radius: 12px; margin: 1rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
            <h2 style='margin-top: 0; color: #34495e; font-size: 1.5rem;'>📋 Candidate Information</h2>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("#### 👤 Personal Details")
        age = st.number_input("Age", 18, 65, 30, key="age_input")
        gender = st.selectbox("Gender", df["Gender"].unique(), key="gender_select")
        race = st.selectbox("Race", df["Race"].unique(), key="race_select")
        ethnicity = st.selectbox("Ethnicity", df["Ethnicity"].unique(), key="ethnicity_select")
    
    with col2:
        st.markdown("#### 💼 Job Application Details")
        job_role = st.selectbox("Job Role", sorted(role_to_desc.keys()), key="job_role_select")
        user_desc = st.text_area(
            "Candidate Job Description",
            "Experienced in data analysis and ML model building.",
            height=100,
            key="user_desc_textarea"
        )

def get_summary_from_resume(resume_text):
    """Extract and summarize resume using Gemini AI"""
    try:
        # === 1. 设置你的 Google API Key ===
        genai.configure(api_key="AIzaSyDTZbFA22Y-QWAftfKlMpmU-E23If2rHPk")

        # ===  构造Prompt ===
        # Limit resume text to first 1200 characters to avoid token limits
        resume_snippet = resume_text[:3000] if len(resume_text) > 3000 else resume_text
        
        prompt = f"""
    You are an assistant that extracts skills or expertise areas from resumes or technical documents.

    Read the following text and summarize the skills and experience in the following cases format:
    Case1: Proficient in Injury Prevention, Motivation, Nutrition, Health Coaching, Strength Training, 
    with mid-level experience in the field. Holds a Bachelors degree. 
    Holds certifications such as Certified Personal Trainer (CPT) by NASM. 
    Skilled in delivering results and adapting to dynamic environments.

    Case2: Proficient in Social Media, Blogging, Creative Writing, Communication, 
    Editing, with senior-level experience in the field. Holds a Bachelors degree. 
    Holds certifications such as Google Analytics for Beginners Certification. 
    Skilled in delivering results and adapting to dynamic environments.

    Case3: Proficient in Creativity, Adobe Photoshop, Web Design, Photo Editing, 
    Illustrator, with senior-level experience in the field. Holds a Masters degree. 
    Holds certifications such as Google UX Design Certificate. Skilled in delivering 
    results and adapting to dynamic environments.

    Avoid adding extra explanation or commentary.
    Return only one line in that exact format.

    Text:
    {resume_snippet}
    """

        # === Call Gemini Model ===
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)

        # === Return  ===
        if response and response.text:
            return response.text.strip()
        else:
            return None
    except Exception as e:
        st.error(f"Error generating resume summary: {str(e)}")
        return None

def get_improvement_feedback(resume_summary, job_description, job_role, match_probability):
    """Generate improvement suggestions using Gemini AI when candidate doesn't match"""
    try:
        # === 1. Set your Google API Key ===
        genai.configure(api_key="AIzaSyDTZbFA22Y-QWAftfKlMpmU-E23If2rHPk")

        # ===  Construct Prompt ===
        prompt = f"""
    You are a career advisor helping a job candidate improve their resume to better match a job position.

    The candidate's current resume summary:
    {resume_summary}

    The job they applied for: {job_role}
    
    The job description:
    {job_description}

    Current match probability: {match_probability:.2%}

    Based on the resume and job requirements, provide exactly 3 specific, actionable suggestions on how the candidate can improve their resume or skills to be a better match for this position.

    IMPORTANT: Format your response EXACTLY as bullet points using HTML <ul> and <li> tags:
    <ul>
    <li>[First suggestion - be specific and actionable]</li>
    <li>[Second suggestion - be specific and actionable]</li>
    <li>[Third suggestion - be specific and actionable]</li>
    </ul>

    Focus on:
    - Missing skills or qualifications mentioned in the job description
    - Experience level gaps
    - Relevant certifications or training
    - How to better highlight relevant experience

    Be constructive and specific. Avoid generic advice. Return ONLY the HTML bullet list, no additional text or explanation.
    """

        # === 调用Gemini模型 ===
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)

        # === 返回结果文本 ===
        if response and response.text:
            return response.text.strip()
        else:
            return None
    except Exception as e:
        st.error(f"Error generating improvement feedback: {str(e)}")
        return None


# User inputs are already defined above in the User Inputs Section

# Initialize session state for caching
if 'resume_raw' not in st.session_state:
    st.session_state.resume_raw = None
if 'resume_summary' not in st.session_state:
    st.session_state.resume_summary = None
if 'processed_file_id' not in st.session_state:
    st.session_state.processed_file_id = None

# PDF Resume Upload Section
st.markdown("---")
with st.container():
    st.markdown("""
        <div style='background: white; padding: 1.5rem; border-radius: 12px; margin: 1rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
            <h2 style='margin-top: 0; color: #34495e; font-size: 1.5rem;'>📄 Upload Your Resume</h2>
            <p style='color: #7f8c8d; margin-bottom: 1rem;'>Upload your resume as a PDF file. Our AI will automatically extract and analyze your skills and experience.</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file to upload your resume",
        type=['pdf'],
        help="Please upload your resume as a PDF file. The text will be automatically extracted for analysis.",
        label_visibility="collapsed"
    )

# Extract text from PDF and generate summary (only if new file uploaded)
if uploaded_file is not None:
    # Check if this is a new file (by comparing file name and size)
    current_file_id = f"{uploaded_file.name}_{uploaded_file.size}" if uploaded_file.name else str(id(uploaded_file))
    
    # Only process if it's a new file or if we don't have a cached summary
    if st.session_state.processed_file_id != current_file_id or st.session_state.resume_summary is None:
        resume_raw = extract_text_from_pdf(uploaded_file)
        if resume_raw and len(resume_raw.strip()) > 10:
            st.success(f"✅ PDF uploaded successfully! Extracted {len(resume_raw)} characters from your resume.")
            
            # Generate summary using Gemini AI (only for new files)
            with st.spinner("🤖 Processing resume to extract key skills and experience..."):
                resume_summary = get_summary_from_resume(resume_raw)
            
            # Store in session state
            st.session_state.resume_raw = resume_raw
            st.session_state.resume_summary = resume_summary
            st.session_state.processed_file_id = current_file_id
            
            if resume_summary:
                st.success("✅ Resume summary generated successfully!")
        else:
            st.session_state.resume_raw = None
            st.session_state.resume_summary = None
            st.session_state.processed_file_id = None
            st.warning("⚠️ Could not extract sufficient text from PDF. Please ensure your PDF contains readable text.")
    else:
        # Use cached values (file already processed)
        resume_raw = st.session_state.resume_raw
        resume_summary = st.session_state.resume_summary
        if resume_summary:
            st.success("✅ Using previously processed resume summary.")
else:
    # Clear session state when no file is uploaded
    st.session_state.resume_raw = None
    st.session_state.resume_summary = None
    st.session_state.processed_file_id = None
    resume_raw = None
    resume_summary = None
    st.info("ℹ️ **Please upload your resume PDF file above to get accurate match predictions.**")

# Ensure resume_summary is available from session state if not set locally
if resume_summary is None and st.session_state.get('resume_summary'):
    resume_summary = st.session_state.resume_summary

# Job Description Display
standard_desc = role_to_desc[job_role]
with st.container():
    st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); padding: 1.5rem; border-radius: 12px; margin: 1rem 0; border-left: 4px solid #667eea;'>
            <h3 style='margin-top: 0; color: #34495e;'>📘 Standard Job Description for <strong>{job_role}</strong></h3>
            <p style='color: #2c3e50; line-height: 1.6;'>{standard_desc}</p>
        </div>
    """, unsafe_allow_html=True)

# ==========================
# Prediction Section
# ==========================
if st.button("🔮 Predict Match Probability"):
    
    # Check if resume was uploaded and text extracted
    if uploaded_file is None:
        st.warning("⚠️ **Please upload your resume PDF file before making a prediction.**")
        st.stop()
    
    # Get resume summary from session state
    resume_summary = st.session_state.get('resume_summary', None)
    
    if resume_summary is None or len(resume_summary.strip()) < 10:
        st.error("❌ **Could not extract sufficient text from the PDF. Please ensure your PDF contains readable text and try uploading again.**")
        st.stop()

    # Use the AI-generated summary for prediction
    resume = resume_summary
    full_desc = user_desc + " " + standard_desc

    # === 1. BERT embeddings (correct shape) ===
    resume_emb = bert_model.encode([resume], convert_to_numpy=True)[0]   # shape: (384,)
    job_emb = bert_model.encode([full_desc], convert_to_numpy=True)[0]  # shape: (384,)

    # concat text features → becomes (768,)
    X_text = np.hstack([resume_emb, job_emb]).reshape(1, -1)
    # NOW shape is (1, 768)  <-- FIXED

    # === 2. semantic similarity ===
    sim = cosine_similarity([resume_emb], [job_emb])[0][0]

    # === 3. tabular features ===
    df_input = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Race": race,
        "Ethnicity": ethnicity,
        "Job Roles": job_role,
        "semantic_similarity": sim
    }])

    X_tab = preprocessor.transform(df_input).toarray()  # shape: (1, N)

    # === 4. final feature ===
    X_final = np.hstack([X_text, X_tab])   # both are 2D → SAFE

    # st.write(f"Model expects features: {model.n_features_in_}")
    # st.write(f"App computed features: {X_final.shape[1]}")

    # === 5. predict ===
    y_pred = model.predict(X_final)[0]
    y_proba = model.predict_proba(X_final)[0][1]

    if y_pred == 1:
        st.success(f"✅ Suitable (Match Probability: {y_proba:.2%})")
    else:
        st.error(f"❌ Not Suitable (Match Probability: {y_proba:.2%})")
        
        # Generate improvement feedback for non-matching candidates
        st.markdown("---")
        st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 12px; margin: 1.5rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
                <h3 style='color: #34495e; margin-top: 0;'>💡 Improvement Suggestions</h3>
                <p style='color: #7f8c8d;'>Based on your resume and the job requirements, here are 3 suggestions to improve your match:</p>
            </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("🤖 Generating personalized improvement suggestions..."):
            feedback = get_improvement_feedback(
                resume_summary=resume_summary,
                job_description=full_desc,
                job_role=job_role,
                match_probability=y_proba
            )
        
        if feedback:
            # Format feedback as bullet points if not already formatted
            # Check if feedback contains HTML list tags
            if '<ul>' not in feedback and '<li>' not in feedback:
                # Convert numbered list or plain text to bullet points
                lines = feedback.split('\n')
                bullet_points = []
                for line in lines:
                    line = line.strip()
                    if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                        # Remove numbering and convert to bullet
                        clean_line = line.lstrip('0123456789.-•) ').strip()
                        if clean_line:
                            bullet_points.append(f"<li>{clean_line}</li>")
                    elif line and len(line) > 10:  # If it's a substantial line
                        bullet_points.append(f"<li>{line}</li>")
                
                if bullet_points:
                    feedback = f"<ul>{''.join(bullet_points)}</ul>"
                else:
                    # Fallback: split by periods and create bullets
                    sentences = [s.strip() for s in feedback.split('.') if s.strip()]
                    if len(sentences) >= 3:
                        feedback = f"<ul><li>{sentences[0]}.</li><li>{sentences[1]}.</li><li>{sentences[2]}.</li></ul>"
            
            # Display feedback in a nicely formatted container with bullet points
            st.markdown(f"""
                <div style='background: #fff9e6; padding: 1.5rem; border-radius: 12px; margin: 1rem 0; border-left: 4px solid #ffc107;'>
                    <div style='color: #2c3e50; line-height: 1.8; font-size: 1.05rem;'>
                        {feedback}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Could not generate improvement suggestions at this time. Please try again.")

    # Probability visualization
    st.markdown("---")
    st.markdown("""
        <div style='background: white; padding: 1.5rem; border-radius: 12px; margin: 1rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
            <h3 style='color: #34495e; margin-top: 0;'>📊 Match Probability Breakdown</h3>
        </div>
    """, unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#dc3545', '#28a745']
    bars = ax.bar(["Not Match", "Match"], model.predict_proba(X_final)[0], color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    ax.set_ylabel('Probability', fontsize=12, fontweight=500, color='#2c3e50')
    ax.set_title('Match Probability Distribution', fontsize=14, fontweight=600, color='#34495e', pad=20)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, model.predict_proba(X_final)[0])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{prob:.1%}',
                ha='center', va='bottom', fontsize=12, fontweight=600, color='#2c3e50')
    
    plt.tight_layout()
    st.pyplot(fig)

# ==========================
# Statistics Section (Collapsible)
# ==========================
with st.expander("📊 View Historical Statistics & Analytics", expanded=False):
    st.header("📈 Historical Job Applications & Match Statistics")
    
    # Calculate overall statistics
    total_applications = len(df)
    total_matches = df["Best Match"].sum()
    match_rate = (total_matches / total_applications) * 100
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Applications", f"{total_applications:,}")
    with col2:
        st.metric("Total Matches", f"{total_matches:,}")
    with col3:
        st.metric("Overall Match Rate", f"{match_rate:.1f}%")
    with col4:
        st.metric("Unique Job Roles", f"{df['Job Roles'].nunique()}")
    
    st.divider()
    
    # Create tabs for different statistics
    tab_names = ["📋 Job Roles", "👥 Demographics", "📊 Age Analysis", "📈 Match Trends"]
    if postings_by_sector is not None or postings_total_us is not None or salary_overview is not None:
        tab_names.extend(["💼 Job Postings", "💰 Salary Overview"])
    
    tabs = st.tabs(tab_names)
    tab1, tab2, tab3, tab4 = tabs[0], tabs[1], tabs[2], tabs[3]
    if len(tabs) > 4:
        tab5, tab6 = tabs[4], tabs[5] if len(tabs) > 5 else None
    
    with tab1:
        st.subheader("Job Role Statistics")
        
        # Job role distribution and match rates
        role_stats = df.groupby("Job Roles").agg({
            "Best Match": ["count", "sum", "mean"]
        }).reset_index()
        role_stats.columns = ["Job Role", "Total Applications", "Matches", "Match Rate"]
        role_stats["Match Rate"] = (role_stats["Match Rate"] * 100).round(2)
        role_stats = role_stats.sort_values("Total Applications", ascending=False)
        
        st.dataframe(role_stats, use_container_width=True, hide_index=True)
        
        # Visualization: Top 10 job roles by application count
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        top_roles = role_stats.head(10)
        ax1.barh(top_roles["Job Role"], top_roles["Total Applications"], color="steelblue")
        ax1.set_xlabel("Number of Applications")
        ax1.set_title("Top 10 Job Roles by Application Volume")
        ax1.invert_yaxis()
        
        # Match rate by job role (top 10 by volume)
        ax2.barh(top_roles["Job Role"], top_roles["Match Rate"], color="green", alpha=0.7)
        ax2.set_xlabel("Match Rate (%)")
        ax2.set_title("Match Rate by Job Role (Top 10)")
        ax2.invert_yaxis()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Pie chart for job role distribution
        fig2, ax = plt.subplots(figsize=(10, 8))
        role_counts = df["Job Roles"].value_counts().head(10)
        ax.pie(role_counts.values, labels=role_counts.index, autopct='%1.1f%%', startangle=90)
        ax.set_title("Job Role Distribution (Top 10)")
        st.pyplot(fig2)
    
    with tab2:
        st.subheader("Demographics Statistics")
        
        # Gender statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Gender Distribution**")
            gender_stats = df.groupby("Gender").agg({
                "Best Match": ["count", "sum", "mean"]
            }).reset_index()
            gender_stats.columns = ["Gender", "Count", "Matches", "Match Rate"]
            gender_stats["Match Rate"] = (gender_stats["Match Rate"] * 100).round(2)
            st.dataframe(gender_stats, use_container_width=True, hide_index=True)
            
            # Gender visualization
            fig, ax = plt.subplots(figsize=(8, 6))
            gender_counts = df["Gender"].value_counts()
            ax.bar(gender_counts.index, gender_counts.values, color=["skyblue", "lightcoral"])
            ax.set_ylabel("Count")
            ax.set_title("Applications by Gender")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with col2:
            st.write("**Race Distribution**")
            race_stats = df.groupby("Race").agg({
                "Best Match": ["count", "sum", "mean"]
            }).reset_index()
            race_stats.columns = ["Race", "Count", "Matches", "Match Rate"]
            race_stats["Match Rate"] = (race_stats["Match Rate"] * 100).round(2)
            race_stats = race_stats.sort_values("Count", ascending=False)
            st.dataframe(race_stats, use_container_width=True, hide_index=True)
            
            # Race visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            race_counts = df["Race"].value_counts()
            ax.barh(race_counts.index, race_counts.values, color="teal")
            ax.set_xlabel("Count")
            ax.set_title("Applications by Race")
            st.pyplot(fig)
        
        # Ethnicity statistics
        st.write("**Ethnicity Statistics**")
        ethnicity_stats = df.groupby("Ethnicity").agg({
            "Best Match": ["count", "sum", "mean"]
        }).reset_index()
        ethnicity_stats.columns = ["Ethnicity", "Count", "Matches", "Match Rate"]
        ethnicity_stats["Match Rate"] = (ethnicity_stats["Match Rate"] * 100).round(2)
        ethnicity_stats = ethnicity_stats.sort_values("Count", ascending=False).head(15)
        st.dataframe(ethnicity_stats, use_container_width=True, hide_index=True)
        
        # Match rate by demographics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        gender_match = df.groupby("Gender")["Best Match"].mean() * 100
        ax1.bar(gender_match.index, gender_match.values, color=["skyblue", "lightcoral"])
        ax1.set_ylabel("Match Rate (%)")
        ax1.set_title("Match Rate by Gender")
        ax1.set_ylim([0, 100])
        
        race_match = df.groupby("Race")["Best Match"].mean() * 100
        ax2.barh(race_match.index, race_match.values, color="teal")
        ax2.set_xlabel("Match Rate (%)")
        ax2.set_title("Match Rate by Race")
        ax2.set_xlim([0, 100])
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab3:
        st.subheader("Age Analysis")
        
        # Age statistics
        age_stats = df.groupby(pd.cut(df["Age"], bins=[18, 25, 30, 35, 40, 45, 50, 55, 65], 
                                      labels=["18-25", "26-30", "31-35", "36-40", "41-45", "46-50", "51-55", "56-65"])).agg({
            "Best Match": ["count", "sum", "mean"]
        }).reset_index()
        age_stats.columns = ["Age Group", "Count", "Matches", "Match Rate"]
        age_stats["Match Rate"] = (age_stats["Match Rate"] * 100).round(2)
        st.dataframe(age_stats, use_container_width=True, hide_index=True)
        
        # Age distribution visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram of age distribution
        ax1.hist(df["Age"], bins=20, color="steelblue", edgecolor="black", alpha=0.7)
        ax1.set_xlabel("Age")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Age Distribution of Applicants")
        ax1.axvline(df["Age"].mean(), color="red", linestyle="--", label=f"Mean: {df['Age'].mean():.1f}")
        ax1.legend()
        
        # Match rate by age group
        age_groups = pd.cut(df["Age"], bins=[18, 25, 30, 35, 40, 45, 50, 55, 65],
                           labels=["18-25", "26-30", "31-35", "36-40", "41-45", "46-50", "51-55", "56-65"])
        age_match_rate = df.groupby(age_groups)["Best Match"].mean() * 100
        ax2.bar(range(len(age_match_rate)), age_match_rate.values, color="green", alpha=0.7)
        ax2.set_xticks(range(len(age_match_rate)))
        ax2.set_xticklabels(age_match_rate.index, rotation=45)
        ax2.set_ylabel("Match Rate (%)")
        ax2.set_title("Match Rate by Age Group")
        ax2.set_ylim([0, 100])
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Age", f"{df['Age'].mean():.1f} years")
        with col2:
            st.metric("Median Age", f"{df['Age'].median():.1f} years")
        with col3:
            st.metric("Age Range", f"{df['Age'].min()}-{df['Age'].max()} years")
    
    with tab4:
        st.subheader("Match Trends & Insights")
        
        # Cross-analysis: Job Role vs Demographics
        st.write("**Match Rate: Job Role × Gender**")
        cross_tab = pd.crosstab(df["Job Roles"], df["Gender"], df["Best Match"], aggfunc="mean") * 100
        cross_tab = cross_tab.round(2)
        st.dataframe(cross_tab, use_container_width=True)
        
        # Top and bottom performing job roles
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top 5 Job Roles by Match Rate**")
            top_match_roles = role_stats.nlargest(5, "Match Rate")[["Job Role", "Match Rate", "Total Applications"]]
            st.dataframe(top_match_roles, use_container_width=True, hide_index=True)
        
        with col2:
            st.write("**Bottom 5 Job Roles by Match Rate**")
            bottom_match_roles = role_stats.nsmallest(5, "Match Rate")[["Job Role", "Match Rate", "Total Applications"]]
            st.dataframe(bottom_match_roles, use_container_width=True, hide_index=True)
        
        # Heatmap: Match rate by Job Role and Gender
        fig, ax = plt.subplots(figsize=(14, 8))
        top_10_roles = role_stats.head(10)["Job Role"].tolist()
        heatmap_data = df[df["Job Roles"].isin(top_10_roles)].groupby(["Job Roles", "Gender"])["Best Match"].mean().unstack(fill_value=0) * 100
        sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax, cbar_kws={'label': 'Match Rate (%)'})
        ax.set_title("Match Rate Heatmap: Top 10 Job Roles × Gender")
        ax.set_xlabel("Gender")
        ax.set_ylabel("Job Role")
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        st.pyplot(fig)
        
        # Summary insights
        st.write("**Key Insights**")
        insights = [
            f"• The overall match rate across all applications is **{match_rate:.1f}%**",
            f"• There are **{df['Job Roles'].nunique()}** unique job roles in the dataset",
            f"• The most popular job role is **{role_stats.iloc[0]['Job Role']}** with {role_stats.iloc[0]['Total Applications']} applications",
            f"• The job role with the highest match rate is **{role_stats.nlargest(1, 'Match Rate').iloc[0]['Job Role']}** ({role_stats.nlargest(1, 'Match Rate').iloc[0]['Match Rate']:.1f}%)",
            f"• Average applicant age is **{df['Age'].mean():.1f}** years",
        ]
        for insight in insights:
            st.write(insight)
    
    # Job Postings Tab
    if len(tabs) > 4 and tab5 is not None:
        with tab5:
            st.subheader("💼 Job Postings Market Analysis")
            
            if postings_total_us is not None:
                st.write("**Total US Job Postings Over Time**")
                
                # Key metrics
                latest_month = postings_total_us['month'].max()
                latest_data = postings_total_us[postings_total_us['month'] == latest_month].iloc[0]
                prev_month_data = postings_total_us[postings_total_us['month'] < latest_month].iloc[-1] if len(postings_total_us[postings_total_us['month'] < latest_month]) > 0 else None
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Latest Month", latest_month.strftime("%b %Y") if pd.notna(latest_month) else "N/A")
                with col2:
                    latest_postings = int(latest_data['active_postings_sa']) if pd.notna(latest_data['active_postings_sa']) else int(latest_data['active_postings_nsa'])
                    prev_postings = int(prev_month_data['active_postings_sa']) if prev_month_data is not None and pd.notna(prev_month_data['active_postings_sa']) else (int(prev_month_data['active_postings_nsa']) if prev_month_data is not None else None)
                    delta = (latest_postings - prev_postings) if prev_postings is not None else None
                    st.metric("Active Postings (SA)", f"{latest_postings:,}", delta=delta if delta is not None else None)
                with col3:
                    avg_postings = int(postings_total_us['active_postings_sa'].mean()) if 'active_postings_sa' in postings_total_us.columns else int(postings_total_us['active_postings_nsa'].mean())
                    st.metric("Average Postings", f"{avg_postings:,}")
                with col4:
                    max_postings = int(postings_total_us['active_postings_sa'].max()) if 'active_postings_sa' in postings_total_us.columns else int(postings_total_us['active_postings_nsa'].max())
                    st.metric("Peak Postings", f"{max_postings:,}")
                
                # Time series plot
                fig, ax = plt.subplots(figsize=(14, 6))
                if 'active_postings_sa' in postings_total_us.columns:
                    ax.plot(postings_total_us['month'], postings_total_us['active_postings_sa'], 
                           label='Seasonally Adjusted', linewidth=2, color='steelblue')
                ax.plot(postings_total_us['month'], postings_total_us['active_postings_nsa'], 
                       label='Not Seasonally Adjusted', linewidth=2, color='orange', alpha=0.7)
                ax.set_xlabel("Month")
                ax.set_ylabel("Number of Active Postings")
                ax.set_title("Total US Job Postings Over Time")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Recent trend table
                st.write("**Recent Job Postings Trend**")
                recent_data = postings_total_us.tail(12).copy()
                recent_data['month'] = recent_data['month'].dt.strftime('%Y-%m')
                recent_data = recent_data[['month', 'active_postings_nsa', 'active_postings_sa']]
                recent_data.columns = ['Month', 'Postings (NSA)', 'Postings (SA)']
                recent_data['Postings (NSA)'] = recent_data['Postings (NSA)'].apply(lambda x: f"{int(x):,}")
                recent_data['Postings (SA)'] = recent_data['Postings (SA)'].apply(lambda x: f"{int(x):,}")
                st.dataframe(recent_data, use_container_width=True, hide_index=True)
            
            if postings_by_sector is not None:
                st.divider()
                st.write("**Job Postings by Industry Sector**")
                
                # Latest month sector analysis
                latest_sector_month = postings_by_sector['month'].max()
                latest_sectors = postings_by_sector[postings_by_sector['month'] == latest_sector_month].copy()
                latest_sectors = latest_sectors.sort_values('active_postings_sa', ascending=False)
                
                # Top sectors
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Top 10 Sectors (Latest: {latest_sector_month.strftime('%b %Y')})**")
                    top_sectors = latest_sectors.head(10)[['naics2d_name', 'active_postings_sa']].copy()
                    top_sectors.columns = ['Sector', 'Active Postings']
                    top_sectors['Active Postings'] = top_sectors['Active Postings'].apply(lambda x: f"{int(x):,}")
                    st.dataframe(top_sectors, use_container_width=True, hide_index=True)
                
                with col2:
                    # Sector distribution pie chart
                    fig, ax = plt.subplots(figsize=(10, 8))
                    top_10_sectors = latest_sectors.head(10)
                    ax.pie(top_10_sectors['active_postings_sa'], labels=top_10_sectors['naics2d_name'], 
                          autopct='%1.1f%%', startangle=90)
                    ax.set_title(f"Top 10 Sectors by Job Postings ({latest_sector_month.strftime('%b %Y')})")
                    st.pyplot(fig)
                
                # Sector trends over time
                st.write("**Sector Trends Over Time (Top 5 Sectors)**")
                top_5_sector_names = latest_sectors.head(5)['naics2d_name'].tolist()
                sector_trends = postings_by_sector[postings_by_sector['naics2d_name'].isin(top_5_sector_names)]
                
                fig, ax = plt.subplots(figsize=(14, 8))
                for sector in top_5_sector_names:
                    sector_data = sector_trends[sector_trends['naics2d_name'] == sector].sort_values('month')
                    ax.plot(sector_data['month'], sector_data['active_postings_sa'], 
                           label=sector, linewidth=2, marker='o', markersize=3)
                ax.set_xlabel("Month")
                ax.set_ylabel("Active Postings (Seasonally Adjusted)")
                ax.set_title("Job Postings Trends: Top 5 Sectors")
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Sector comparison table
                st.write("**Sector Comparison (Latest Month)**")
                sector_comparison = latest_sectors[['naics2d_name', 'active_postings_nsa', 'active_postings_sa']].copy()
                sector_comparison.columns = ['Sector', 'Postings (NSA)', 'Postings (SA)']
                sector_comparison['Postings (NSA)'] = sector_comparison['Postings (NSA)'].apply(lambda x: f"{int(x):,}")
                sector_comparison['Postings (SA)'] = sector_comparison['Postings (SA)'].apply(lambda x: f"{int(x):,}")
                st.dataframe(sector_comparison, use_container_width=True, hide_index=True)
    
    # Salary Overview Tab
    if len(tabs) > 5 and tab6 is not None and salary_overview is not None:
        with tab6:
            st.subheader("💰 Salary Overview by Occupation")
            
            # Clean salary data - remove dollar signs and commas, convert to numeric
            salary_cols = ['Oct 2024', 'Aug 2025', 'Sep 2025', 'Oct 2025']
            for col in salary_cols:
                if col in salary_overview.columns:
                    try:
                        salary_overview[col + '_num'] = salary_overview[col].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
                    except:
                        # If conversion fails, try alternative method
                        salary_overview[col + '_num'] = pd.to_numeric(salary_overview[col].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
            
            # Overall statistics
            total_row = salary_overview[salary_overview['soc2d_name'] == 'Total US']
            if len(total_row) > 0:
                total_salary = total_row.iloc[0]
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Oct 2024 Avg Salary", f"${total_salary['Oct 2024']}")
                with col2:
                    st.metric("Oct 2025 Avg Salary", f"${total_salary['Oct 2025']}")
                with col3:
                    yoy_change = total_salary['Pct change YoY (Oct 2024 - Oct 2025)'] * 100
                    st.metric("Year-over-Year Change", f"{yoy_change:.2f}%", delta=f"{yoy_change:.2f}%")
                with col4:
                    mom_change = total_salary['Pct change (Sep 2025 - Oct 2025)'] * 100
                    st.metric("Month-over-Month Change", f"{mom_change:.2f}%", delta=f"{mom_change:.2f}%")
            
            st.divider()
            
            # Filter out total row for detailed analysis
            salary_detail = salary_overview[salary_overview['soc2d_name'] != 'Total US'].copy()
            
            # Top and bottom occupations by salary
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top 10 Occupations by Salary (Oct 2025)**")
                top_salaries = salary_detail.nlargest(10, 'Oct 2025_num')[['soc2d_name', 'Oct 2025', 'Pct change YoY (Oct 2024 - Oct 2025)']].copy()
                top_salaries.columns = ['Occupation', 'Salary (Oct 2025)', 'YoY Change']
                top_salaries['YoY Change'] = (top_salaries['YoY Change'] * 100).round(2).astype(str) + '%'
                st.dataframe(top_salaries, use_container_width=True, hide_index=True)
            
            with col2:
                st.write("**Bottom 10 Occupations by Salary (Oct 2025)**")
                bottom_salaries = salary_detail.nsmallest(10, 'Oct 2025_num')[['soc2d_name', 'Oct 2025', 'Pct change YoY (Oct 2024 - Oct 2025)']].copy()
                bottom_salaries.columns = ['Occupation', 'Salary (Oct 2025)', 'YoY Change']
                bottom_salaries['YoY Change'] = (bottom_salaries['YoY Change'] * 100).round(2).astype(str) + '%'
                st.dataframe(bottom_salaries, use_container_width=True, hide_index=True)
            
            # Salary distribution visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Top 15 salaries bar chart
            top_15 = salary_detail.nlargest(15, 'Oct 2025_num')
            ax1.barh(range(len(top_15)), top_15['Oct 2025_num'], color='steelblue')
            ax1.set_yticks(range(len(top_15)))
            ax1.set_yticklabels(top_15['soc2d_name'], fontsize=9)
            ax1.set_xlabel("Salary ($)")
            ax1.set_title("Top 15 Occupations by Salary (Oct 2025)")
            ax1.invert_yaxis()
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Salary change YoY
            top_growers = salary_detail.nlargest(10, 'Pct change YoY (Oct 2024 - Oct 2025)')
            ax2.barh(range(len(top_growers)), top_growers['Pct change YoY (Oct 2024 - Oct 2025)'] * 100, color='green', alpha=0.7)
            ax2.set_yticks(range(len(top_growers)))
            ax2.set_yticklabels(top_growers['soc2d_name'], fontsize=9)
            ax2.set_xlabel("Year-over-Year Change (%)")
            ax2.set_title("Top 10 Salary Growth (YoY)")
            ax2.invert_yaxis()
            ax2.axvline(x=0, color='red', linestyle='--', linewidth=1)
            ax2.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Salary trends over time
            st.write("**Salary Trends: Top 5 Occupations**")
            top_5_occ = salary_detail.nlargest(5, 'Oct 2025_num')
            
            fig, ax = plt.subplots(figsize=(14, 6))
            x_pos = ['Oct 2024', 'Aug 2025', 'Sep 2025', 'Oct 2025']
            x_numeric = range(len(x_pos))
            
            for idx, row in top_5_occ.iterrows():
                salaries = [row['Oct 2024_num'], row['Aug 2025_num'], row['Sep 2025_num'], row['Oct 2025_num']]
                ax.plot(x_numeric, salaries, marker='o', linewidth=2, label=row['soc2d_name'], markersize=6)
            
            ax.set_xticks(x_numeric)
            ax.set_xticklabels(x_pos)
            ax.set_ylabel("Salary ($)")
            ax.set_title("Salary Trends: Top 5 Highest-Paying Occupations")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Full salary table
            st.write("**Complete Salary Overview**")
            salary_table = salary_detail[['soc2d_name', 'Oct 2024', 'Oct 2025', 'Pct change YoY (Oct 2024 - Oct 2025)', 'Pct change (Sep 2025 - Oct 2025)']].copy()
            salary_table.columns = ['Occupation', 'Oct 2024', 'Oct 2025', 'YoY Change (%)', 'MoM Change (%)']
            salary_table['YoY Change (%)'] = (salary_table['YoY Change (%)'] * 100).round(2)
            salary_table['MoM Change (%)'] = (salary_table['MoM Change (%)'] * 100).round(2)
            # Sort by numeric salary value
            if 'Oct 2025_num' in salary_detail.columns:
                salary_table = salary_table.assign(sort_key=salary_detail['Oct 2025_num'].values)
                salary_table = salary_table.sort_values('sort_key', ascending=False).drop('sort_key', axis=1)
            st.dataframe(salary_table, use_container_width=True, hide_index=True)
