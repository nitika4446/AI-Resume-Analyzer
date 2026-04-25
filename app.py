import streamlit as st
import pandas as pd
import pdfplumber
import re
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# -----------------------------
# Load Training Dataset
# -----------------------------
@st.cache_data
def load_dataset():
    df = pd.read_csv("UpdatedResumeDataSet.csv")
    return df

df = load_dataset()

# -----------------------------
# Train Resume Category Model
# -----------------------------
X = df["Resume"]
y = df["Category"]

vectorizer = TfidfVectorizer(stop_words="english")
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

# -----------------------------
# PDF Text Extraction
# -----------------------------
def extract_pdf_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


# -----------------------------
# CSV Text Extraction
# -----------------------------
def extract_csv_text(file):
    df = pd.read_csv(file)

    # Assuming resume text is stored in 'Resume' column
    if "Resume" in df.columns:
        text = " ".join(df["Resume"].astype(str))
    else:
        text = " ".join(df.astype(str).values.flatten())

    return text


# -----------------------------
# Skill Extraction
# -----------------------------
def extract_skills(text):
    skills_list = [
        "python", "sql", "machine learning",
        "data science", "excel", "power bi",
        "tableau", "java", "streamlit"
    ]

    found_skills = []

    for skill in skills_list:
        if re.search(r"\b" + re.escape(skill) + r"\b", text.lower()):
            found_skills.append(skill)

    return found_skills


# -----------------------------
# Resume Score
# -----------------------------
def resume_score(text):
    score = 0
    sections = ["education", "skills", "experience", "projects"]

    for sec in sections:
        if sec in text.lower():
            score += 25

    return score


# -----------------------------
# Keyword Chart
# -----------------------------
def keyword_chart(text):
    words = re.findall(r'\w+', text.lower())

    stopwords = {"the", "and", "is", "to", "of", "for"}

    filtered_words = [w for w in words if w not in stopwords]

    word_count = Counter(filtered_words).most_common(10)

    if word_count:
        labels = [x[0] for x in word_count]
        values = [x[1] for x in word_count]

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        plt.xticks(rotation=45)
        st.pyplot(fig)


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📄 AI Resume Analysis System")

uploaded_file = st.file_uploader(
    "Upload Resume",
    type=["pdf", "csv"]
)

if uploaded_file:

    file_type = uploaded_file.name.split(".")[-1]

    if file_type == "pdf":
        resume_text = extract_pdf_text(uploaded_file)

    elif file_type == "csv":
        resume_text = extract_csv_text(uploaded_file)

    st.subheader("Extracted Resume Text")
    st.text_area("Resume Content", resume_text, height=250)

    # Predict category
    transformed_text = vectorizer.transform([resume_text])
    prediction = model.predict(transformed_text)[0]

    st.subheader("Predicted Job Category")
    st.success(prediction)

    # Skills extraction
    skills = extract_skills(resume_text)

    st.subheader("Detected Skills")
    st.write(skills)

    # Resume score
    score = resume_score(resume_text)

    st.subheader("Resume Score")
    st.metric("Score", f"{score}/100")

    # Keyword chart
    st.subheader("Top Resume Keywords")
    keyword_chart(resume_text)