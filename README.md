# 🧠 HireLens — AI Resume Screener

An AI-powered resume screening tool that compares multiple resumes against a job description and provides ranked results with detailed feedback, ATS keyword analysis, and improvement suggestions.

**Stack:** LangChain · LLaMA 3.3 70B (Groq) · Streamlit · Pydantic · PyPDF

---

## ✨ Features

- **Multi-resume upload** — screen up to 5 resumes at once
- **Structured AI scoring** — Overall + 4 sub-scores (Skills, Experience, Education, Projects)
- **Hire recommendation** — Strong Hire / Hire / Maybe / No Hire
- **ATS keyword detection** — shows which JD keywords appear in each resume
- **Gap analysis** — missing skills & specific improvement suggestions
- **Compare tab** — side-by-side score breakdown table + bar chart
- **Session history** — tracks all screenings in the current session
- **JSON export** — download full results for reporting

---

## 🚀 Local Setup

```bash
# 1. Clone and enter project
git clone <your-repo-url>
cd resume-screener

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variable
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (free at https://console.groq.com)

# 5. Run
streamlit run app.py
```

---

## ☁️ Deploy to Streamlit Community Cloud (Free)

**Step 1 — Push to GitHub**
```bash
git init
git add .
git commit -m "Initial commit — ResumeIQ"
git remote add origin https://github.com/YOUR_USERNAME/resume-screener.git
git push -u origin main
```

**Step 2 — Go to [share.streamlit.io](https://share.streamlit.io)**
1. Click **New app**
2. Connect your GitHub repo
3. Set **Main file path** to `app.py`
4. Click **Advanced settings** → **Secrets**
5. Add your secret:
   ```toml
   GROQ_API_KEY = "gsk_your_actual_key_here"
   ```
6. Click **Deploy** — done! 🎉


---

```

---

## 📁 Project Structure

```
resume-screener/
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .streamlit/
│   └── config.toml         # Streamlit theme & server config
└── README.md
```

---

## 🔑 Getting a Free Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up / Log in
3. Go to **API Keys** → **Create API Key**
4. Copy it to your `.env` or Streamlit Secrets

---

## 🛠️ Built With

| Tech | Purpose |
|---|---|
| Streamlit | Frontend UI |
| LangChain | LLM orchestration |
| Groq + LLaMA 3.3 70B | Fast, free LLM inference |
| PyPDF | PDF text extraction |
| Pydantic | Structured output validation |
