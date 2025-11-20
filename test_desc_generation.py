import google.generativeai as genai
from PyPDF2 import PdfReader

# === 1. 设置你的 Google API Key ===
genai.configure(api_key="AIzaSyDTZbFA22Y-QWAftfKlMpmU-E23If2rHPk")

# === 2. 从PDF提取文字 ===
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

pdf_text = extract_text_from_pdf("data_sample/resume_sample.pdf")

# === 3. 构造Prompt ===
prompt = f"""
You are an assistant that extracts skills or expertise areas from resumes or technical documents.

Read the following text and summarize the skills in this fixed format:
Proficient in <skill1>, <skill2>, <skill3>, ...

If a skill is not directly mentioned, infer it only if it is clearly implied by the text.
Avoid adding extra explanation or commentary.
Return only one line in that exact format.

Text:
{pdf_text[:1200]}
"""

# === 4. 调用Gemini模型 ===
model = genai.GenerativeModel("gemini-2.5-flash")
response = model.generate_content(prompt)

# === 5. 输出结果 ===
print("=== Extracted Skills ===")
print(response.text)
