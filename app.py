import streamlit as st
import os
import tempfile
import json
import time
from datetime import datetime
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ConfigDict

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HireLens — AI Resume Screener",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp {
    background: #0d0f14;
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111318 !important;
    border-right: 1px solid #1e2230;
}
[data-testid="stSidebar"] * {
    color: #c8d0e0 !important;
}

/* Header */
.resume-iq-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6366f1 0%, #a78bfa 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.resume-iq-sub {
    color: #64748b;
    font-size: 1rem;
    margin-bottom: 2rem;
}

/* Cards */
.card {
    background: #161a24;
    border: 1px solid #1e2638;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

/* Score Badge */
.score-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 52px;
    height: 52px;
    border-radius: 50%;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    margin-right: 1rem;
    flex-shrink: 0;
}
.score-high   { background: #052e1c; color: #4ade80; border: 2px solid #16a34a; }
.score-mid    { background: #1c1a05; color: #fbbf24; border: 2px solid #d97706; }
.score-low    { background: #2e0505; color: #f87171; border: 2px solid #dc2626; }

/* Rank card */
.rank-card {
    background: #161a24;
    border: 1px solid #1e2638;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}
.rank-card:hover { border-color: #6366f1; }
.rank-card-top   { border-left: 4px solid #6366f1; }

/* Progress bar custom */
.progress-wrap { margin: 0.3rem 0 0.8rem 0; }
.progress-label { font-size: 0.8rem; color: #94a3b8; margin-bottom: 0.2rem; display: flex; justify-content: space-between; }
.progress-bar-bg { background: #1e2638; border-radius: 99px; height: 7px; width: 100%; }
.progress-bar-fill { height: 7px; border-radius: 99px; }

/* Tag */
.tag {
    display: inline-block;
    padding: 0.22rem 0.7rem;
    border-radius: 99px;
    font-size: 0.73rem;
    font-weight: 500;
    margin: 0.15rem 0.15rem 0 0;
}
.tag-green { background: #052e1c; color: #4ade80; border: 1px solid #166534; }
.tag-red   { background: #2e0505; color: #f87171; border: 1px solid #991b1b; }
.tag-blue  { background: #05182e; color: #60a5fa; border: 1px solid #1e40af; }

/* Section heading */
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.7rem;
}

/* Strealit button override */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.65rem 2rem !important;
    font-size: 0.95rem !important;
    transition: opacity 0.2s !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Metric */
.big-metric {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #6366f1;
}
.metric-label { font-size: 0.8rem; color: #64748b; margin-top: -0.4rem; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #161a24;
    border: 1.5px dashed #2d3748;
    border-radius: 10px;
    padding: 0.5rem;
}

/* Expander */
.streamlit-expanderHeader {
    background: #161a24 !important;
    border: 1px solid #1e2638 !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* Input & Textarea */
.stTextArea textarea, .stTextInput input {
    background: #161a24 !important;
    border: 1px solid #2d3748 !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}

/* Alert */
.stAlert { border-radius: 10px; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 99px; }

/* divider */
hr { border-color: #1e2638; }

/* spinner text */
.stSpinner > div > div { border-top-color: #6366f1 !important; }

/* Tab */
.stTabs [data-baseweb="tab-list"] { background: #111318; border-radius: 10px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent; border-radius: 8px; color: #64748b !important; }
.stTabs [aria-selected="true"] { background: #1e2638 !important; color: #e2e8f0 !important; }

</style>
""", unsafe_allow_html=True)


# ─── LLM & Pydantic ─────────────────────────────────────────────────────────
@st.cache_resource
def get_llm():
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("api_key")
    if not api_key:
        st.error("⚠️ GROQ_API_KEY not found. Please add it to your .env file.")
        st.stop()
    return ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0.4)


class ResumeScore(BaseModel):
    model_config = ConfigDict(extra="allow")
    score: float = Field(description="Overall ATS score of resume, a number between 1.0 and 10.0")
    candidate_name: str = Field(description="Full name of the candidate")
    current_role: str = Field(description="Current or most recent job title of candidate")
    total_experience_years: float = Field(description="Total years of relevant experience")
    skill_match_score: float = Field(description="Skill match sub-score, a number between 1.0 and 10.0")
    experience_match_score: float = Field(description="Experience match sub-score, a number between 1.0 and 10.0")
    education_match_score: float = Field(description="Education match sub-score, a number between 1.0 and 10.0")
    project_relevance_score: float = Field(description="Project relevance sub-score, a number between 1.0 and 10.0")
    strength: list[str] = Field(description="Top 3–5 strengths relevant to the JD")
    weakness: list[str] = Field(description="Top 3–5 gaps or weaknesses relative to the JD")
    missing_skills: list[str] = Field(description="Key skills from JD that are absent in the resume")
    improvement_suggestions: list[str] = Field(description="Specific, actionable suggestions to improve this resume for the JD")
    ats_keywords_found: list[str] = Field(description="ATS keywords from the JD found in the resume")
    summary: str = Field(description="2–3 sentence recruiter-style summary of the candidate")
    hire_recommendation: str = Field(description="One of: 'Strong Hire', 'Hire', 'Maybe', 'No Hire'")


# ─── Chain ──────────────────────────────────────────────────────────────────
def chain_creator():
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a senior technical recruiter and ATS expert with 15+ years of experience.

Evaluate resumes against a job description using this scoring rubric:
- Skill Match       (40%): Do the candidate's skills align with required/preferred skills in the JD?
- Experience Match  (30%): Is the candidate's work experience relevant in domain, seniority, and duration?
- Education Match   (10%): Does their educational background meet requirements?
- Project Relevance (20%): Are their projects/portfolio relevant to the role?

Rules:
- Be objective and data-driven. Never invent information.
- Score each dimension 1–10, then compute overall score proportionally.
- For missing_skills: only list skills explicitly mentioned in JD but absent in resume.
- For improvement_suggestions: be specific and actionable (e.g., "Add a quantified metric to your internship bullet points").
- For hire_recommendation: use 'Strong Hire' (9-10), 'Hire' (7-8), 'Maybe' (5-6), 'No Hire' (<5).
- ats_keywords_found should list exact keywords from JD present in the resume text.
"""),
        ("human", """
JOB DESCRIPTION:
{jd}

RESUME TEXT:
{resume_info}

Provide a thorough evaluation of this candidate.
""")
    ])
    return prompt | llm.with_structured_output(ResumeScore)


# ─── PDF Processing ──────────────────────────────────────────────────────────
def process_pdfs(temp_paths, filenames):
    results = []
    for path, name in zip(temp_paths, filenames):
        try:
            loader = PyPDFLoader(path)
            pages = loader.load()
            text = " ".join(p.page_content for p in pages).strip()
            if not text:
                st.warning(f"⚠️ Could not extract text from {name}. It may be image-based.")
            results.append({"text": text, "filename": name})
        except Exception as e:
            st.error(f"Error reading {name}: {e}")
    return results


# ─── Score Color Helper ─────────────────────────────────────────────────────
def score_class(score):
    if score >= 8: return "score-high"
    if score >= 5: return "score-mid"
    return "score-low"


def score_color(score):
    if score >= 8: return "#4ade80"
    if score >= 5: return "#fbbf24"
    return "#f87171"


def hire_badge(rec):
    colors = {
        "Strong Hire": ("#052e1c", "#4ade80"),
        "Hire":        ("#05182e", "#60a5fa"),
        "Maybe":       ("#1c1a05", "#fbbf24"),
        "No Hire":     ("#2e0505", "#f87171"),
    }
    bg, fg = colors.get(rec, ("#1e2638", "#94a3b8"))
    return f'<span style="background:{bg};color:{fg};padding:3px 12px;border-radius:99px;font-size:0.78rem;font-weight:600;">{rec}</span>'


def progress_bar_html(label, value, max_val=10, color="#6366f1"):
    pct = (value / max_val) * 100
    return f"""
<div class="progress-wrap">
  <div class="progress-label"><span>{label}</span><span style="color:{color};font-weight:600;">{value}/{max_val}</span></div>
  <div class="progress-bar-bg">
    <div class="progress-bar-fill" style="width:{pct}%;background:{color};"></div>
  </div>
</div>
"""


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 ResumeIQ")
    st.markdown("<hr style='margin:0.5rem 0 1rem 0'>", unsafe_allow_html=True)

    st.markdown("**Upload Options**")
    max_resumes = st.slider("Max resumes to compare", 1, 5, 3)
    st.markdown("---")

    st.markdown("**Model Settings**")
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.4, 0.1,
                            help="Lower = more consistent scores, Higher = more varied")
    st.markdown("---")

    st.markdown("**Export**")
    export_enabled = st.checkbox("Enable JSON export of results", value=True)

    st.markdown("---")
    st.markdown("**About**")
    st.caption("ResumeIQ uses LLaMA 3.3 70B via Groq + LangChain to screen and rank candidates against a JD.")
    st.caption("Built with ❤️ using Streamlit")


# ─── Main ────────────────────────────────────────────────────────────────────
st.markdown('<div class="resume-iq-header">🧠 ResumeIQ</div>', unsafe_allow_html=True)
st.markdown('<div class="resume-iq-sub">AI-powered resume screener — upload resumes, paste a JD, get ranked results with actionable feedback.</div>', unsafe_allow_html=True)

tabs = st.tabs(["📤 Screen Resumes", "📊 Compare View", "📋 History"])

# ──────────── TAB 1: Screen ────────────────────────────────────────────────
with tabs[0]:
    # Input panel — two equal columns side by side
    st.markdown("""
    <style>
    /* Force equal-height columns in the input panel */
    div[data-testid="column"] > div[data-testid="stVerticalBlock"] {
        height: 100%;
    }
    /* Radio button row compact */
    div[data-testid="stRadio"] > label { display: none; }
    div[data-testid="stRadio"] > div { flex-direction: row; gap: 0.5rem; }
    div[data-testid="stRadio"] > div > label {
        background: #1e2638;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 0.3rem 0.9rem;
        font-size: 0.82rem;
        cursor: pointer;
        color: #94a3b8 !important;
    }
    div[data-testid="stRadio"] > div > label[data-checked="true"],
    div[data-testid="stRadio"] input:checked + div {
        background: #2d3748 !important;
        color: #e2e8f0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    col_upload, col_jd = st.columns(2, gap="large")

    with col_upload:
        st.markdown('<div class="section-label">📄 Resume Files (PDF)</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            f"Upload up to {max_resumes} resumes",
            type=["pdf"],
            accept_multiple_files=True,
            help="Only PDF files are supported. Max 10 MB each.",
            key="resume_uploader",
            label_visibility="collapsed",
        )
        if uploaded_files:
            uploaded_files = uploaded_files[:max_resumes]
            st.success(f"✅ {len(uploaded_files)} file(s) ready")
            for f in uploaded_files:
                size_kb = len(f.getvalue()) / 1024
                st.markdown(f'<span class="tag tag-blue">📄 {f.name} &nbsp;({size_kb:.0f} KB)</span>', unsafe_allow_html=True)
        else:
            st.caption(f"Drag & drop or browse — up to {max_resumes} PDFs at once")

    with col_jd:
        st.markdown('<div class="section-label">📋 Job Description</div>', unsafe_allow_html=True)

        example_jd = """We are looking for a Machine Learning Engineer with:
- 2+ years experience with Python, FastAPI, and REST APIs
- Hands-on experience with LangChain, LlamaIndex, or similar LLM frameworks
- Familiarity with RAG pipelines, vector databases (FAISS, Pinecone)
- Experience with React or any frontend framework (bonus)
- Strong understanding of NLP and deep learning fundamentals
- Bachelor's/Master's in CS, AI, or related field"""

        jd_input_method = st.radio(
            "JD source",
            ["✏️ Paste text", "📌 Use example JD"],
            horizontal=True,
            key="jd_method"
        )

        if jd_input_method == "📌 Use example JD":
            job_description = st.text_area(
                "Job Description",
                value=example_jd,
                height=230,
                label_visibility="collapsed",
            )
        else:
            job_description = st.text_area(
                "Job Description",
                height=230,
                placeholder="Paste the full job description here — requirements, skills, responsibilities...",
                label_visibility="collapsed",
            )

    st.markdown("<br>", unsafe_allow_html=True)
    run_col, _ = st.columns([1, 3])
    with run_col:
        run_btn = st.button("🚀 Screen Resumes", width='stretch')

    if run_btn:
        if not uploaded_files:
            st.warning("Please upload at least one PDF resume.")
        elif not job_description.strip():
            st.warning("Please enter or paste a job description.")
        else:
            temp_paths, filenames = [], []
            for f in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.getvalue())
                    temp_paths.append(tmp.name)
                    filenames.append(f.name)

            with st.spinner("Reading resumes and evaluating candidates..."):
                try:
                    chain = chain_creator()
                    resume_data = process_pdfs(temp_paths, filenames)

                    valid = [(r, r["filename"]) for r in resume_data if r["text"]]
                    if not valid:
                        st.error("No readable text found in uploaded PDFs.")
                    else:
                        inputs = [{"jd": job_description, "resume_info": r["text"]} for r, _ in valid]
                        progress_bar = st.progress(0, text="Evaluating candidates...")

                        results_raw = []
                        for idx, inp in enumerate(inputs):
                            r = chain.invoke(inp)
                            r.filename = valid[idx][1]
                            results_raw.append(r)
                            progress_bar.progress((idx + 1) / len(inputs), text=f"Evaluated {idx+1}/{len(inputs)} resumes...")
                            time.sleep(0.3)

                        progress_bar.empty()

                        results_raw.sort(key=lambda x: x.score, reverse=True)
                        st.session_state["results"] = results_raw
                        st.session_state["jd"] = job_description
                        st.session_state["timestamp"] = datetime.now().strftime("%d %b %Y, %I:%M %p")

                        # Append to history
                        if "history" not in st.session_state:
                            st.session_state["history"] = []
                        st.session_state["history"].append({
                            "timestamp": st.session_state["timestamp"],
                            "results": results_raw,
                            "jd_snippet": job_description[:120] + "..."
                        })

                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
                finally:
                    for p in temp_paths:
                        try: os.unlink(p)
                        except: pass

    # ─── Results ────────────────────────────────────────────────────────────
    if "results" in st.session_state:
        results = st.session_state["results"]
        st.markdown("---")
        st.markdown(f'<div class="section-label">Results — {st.session_state.get("timestamp", "")}</div>', unsafe_allow_html=True)

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        avg_score = sum(r.score for r in results) / len(results)
        top_candidate = results[0].candidate_name
        strong_hires = sum(1 for r in results if r.hire_recommendation in ["Strong Hire", "Hire"])

        with m1:
            st.markdown(f'<div class="card"><div class="big-metric">{len(results)}</div><div class="metric-label">Candidates screened</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="card"><div class="big-metric">{avg_score:.1f}</div><div class="metric-label">Average score /10</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="card"><div class="big-metric">{strong_hires}</div><div class="metric-label">Recommended hires</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="card"><div class="big-metric">#{1}</div><div class="metric-label">Top: {top_candidate.split()[0]}</div></div>', unsafe_allow_html=True)

        medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]

        for rank, result in enumerate(results):
            medal = medals[rank] if rank < len(medals) else f"#{rank+1}"
            sc = score_class(result.score)
            color = score_color(result.score)
            top_cls = "rank-card-top" if rank == 0 else ""

            with st.expander(
                f"{medal} {result.candidate_name}  •  Score: {result.score:.1f}/10  •  {result.hire_recommendation}  |  {result.filename}",
                expanded=(rank == 0)
            ):
                # Top row
                h1, h2 = st.columns([2, 1])
                with h1:
                    st.markdown(f"**🧑‍💼 {result.current_role}** · {result.total_experience_years:.1f} yrs experience")
                    st.markdown(f"**📋 Summary:** {result.summary}")
                    st.markdown(hire_badge(result.hire_recommendation), unsafe_allow_html=True)
                with h2:
                    st.markdown(progress_bar_html("Skill Match",      result.skill_match_score,      color="#6366f1"), unsafe_allow_html=True)
                    st.markdown(progress_bar_html("Experience",       result.experience_match_score, color="#8b5cf6"), unsafe_allow_html=True)
                    st.markdown(progress_bar_html("Education",        result.education_match_score,  color="#38bdf8"), unsafe_allow_html=True)
                    st.markdown(progress_bar_html("Project Relevance",result.project_relevance_score,color="#34d399"), unsafe_allow_html=True)

                st.markdown("---")
                c1, c2, c3 = st.columns(3)

                with c1:
                    st.markdown("**✅ Strengths**")
                    for s in result.strength:
                        st.markdown(f'<span class="tag tag-green">✓ {s}</span>', unsafe_allow_html=True)

                with c2:
                    st.markdown("**❌ Gaps**")
                    for w in result.weakness:
                        st.markdown(f'<span class="tag tag-red">✗ {w}</span>', unsafe_allow_html=True)
                    if result.missing_skills:
                        st.markdown("**🔍 Missing Skills**")
                        for sk in result.missing_skills:
                            st.markdown(f'<span class="tag tag-red">⚠ {sk}</span>', unsafe_allow_html=True)

                with c3:
                    st.markdown("**💡 Improvement Suggestions**")
                    for i, sug in enumerate(result.improvement_suggestions, 1):
                        st.markdown(f"**{i}.** {sug}")

                # ATS Keywords
                if result.ats_keywords_found:
                    st.markdown("---")
                    st.markdown("**🔑 ATS Keywords Found**")
                    kw_html = " ".join(f'<span class="tag tag-blue">{kw}</span>' for kw in result.ats_keywords_found)
                    st.markdown(kw_html, unsafe_allow_html=True)

        # Export
        if export_enabled:
            st.markdown("---")
            export_data = []
            for r in results:
                export_data.append({
                    "rank": results.index(r) + 1,
                    "candidate_name": r.candidate_name,
                    "filename": r.filename,
                    "score": r.score,
                    "hire_recommendation": r.hire_recommendation,
                    "current_role": r.current_role,
                    "total_experience_years": r.total_experience_years,
                    "skill_match_score": r.skill_match_score,
                    "experience_match_score": r.experience_match_score,
                    "education_match_score": r.education_match_score,
                    "project_relevance_score": r.project_relevance_score,
                    "strengths": r.strength,
                    "weaknesses": r.weakness,
                    "missing_skills": r.missing_skills,
                    "improvement_suggestions": r.improvement_suggestions,
                    "ats_keywords_found": r.ats_keywords_found,
                    "summary": r.summary,
                })
            json_str = json.dumps({"screened_at": st.session_state.get("timestamp"), "results": export_data}, indent=2)
            st.download_button(
                label="⬇️ Export Results as JSON",
                data=json_str,
                file_name=f"resume_screening_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )


# Comparison
with tabs[1]:
    if "results" not in st.session_state:
        st.info("Run a screening first to see the comparison chart here.")
    else:
        results = st.session_state["results"]
        st.markdown('<div class="section-label">Score Breakdown Comparison</div>', unsafe_allow_html=True)

        import pandas as pd
        data = {
            "Candidate": [r.candidate_name for r in results],
            "Overall": [r.score for r in results],
            "Skills": [r.skill_match_score for r in results],
            "Experience": [r.experience_match_score for r in results],
            "Education": [r.education_match_score for r in results],
            "Projects": [r.project_relevance_score for r in results],
        }
        df = pd.DataFrame(data)
        df.index = range(1, len(df) + 1)  # 1-based rank index, always unique
        st.dataframe(df.style.background_gradient(subset=["Overall","Skills","Experience","Education","Projects"], cmap="RdYlGn", vmin=1, vmax=10), width='stretch')

        st.markdown("<br>", unsafe_allow_html=True)
        st.bar_chart(df.set_index("Candidate")[["Overall", "Skills", "Experience", "Education", "Projects"]])

        st.markdown('<div class="section-label">Side-by-Side Candidate Details</div>', unsafe_allow_html=True)
        cols = st.columns(len(results))
        for col, r in zip(cols, results):
            with col:
                color = score_color(r.score)
                st.markdown(f"**{r.candidate_name}**")
                st.markdown(f'<span style="color:{color};font-size:1.8rem;font-weight:700;">{r.score:.1f}/10</span>', unsafe_allow_html=True)
                st.caption(r.hire_recommendation)
                st.markdown("**Missing:**")
                for sk in r.missing_skills[:3]:
                    st.markdown(f"- {sk}")


#  History 
with tabs[2]:
    if "history" not in st.session_state or not st.session_state["history"]:
        st.info("No screening sessions yet. Run a screening to see history here.")
    else:
        st.markdown('<div class="section-label">Past Screening Sessions (this session)</div>', unsafe_allow_html=True)
        for i, session in enumerate(reversed(st.session_state["history"])):
            with st.expander(f"Session {len(st.session_state['history']) - i} — {session['timestamp']}"):
                st.caption(f"JD: {session['jd_snippet']}")
                for r in sorted(session["results"], key=lambda x: x.score, reverse=True):
                    st.markdown(f"- **{r.candidate_name}** — {r.score:.1f}/10 — {r.hire_recommendation}")
