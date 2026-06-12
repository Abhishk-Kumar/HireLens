#core functionality of the project


from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import streamlit as st

import os
import tempfile

load_dotenv()

llm=ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("api_key"),
    temperature=0.5
)
class ResumeScore(BaseModel):
    filename: str = ""
    score: int =Field(description="It tells the score of resume out of 10")
    candidate_name:str=Field(description="Name of the candidate")
    strength:list[str] =Field(description="Strength of the candidate")
    weakness:list[str] =Field(description="Weakness of the candidate")
    summary: str = Field(description="Summary of the candidate's resume")

def chain_creator():
    prompt = ChatPromptTemplate.from_messages([
    
        ("system", """
        You are an expert ATS and technical recruiter.
        Evaluate based on:
        1. Skill Match (40%)
        2. Experience Match (30%)
        3. Education Match (10%)
        4. Project Relevance (20%)
        Score between 1-10. Be objective. Do not invent information.
        """),
        ("human", """
        JOB DESCRIPTION:
        {jd}

        RESUME:
        {resume_info}

        Evaluate this candidate.
        """)
    ])
    structured_llm=llm.with_structured_output(ResumeScore)
    chain= prompt | structured_llm
    return chain

def pdfProcessor(docs, filenames):
    resumeinfo=[]
    for pdf, filename in zip(docs, filenames):
        loader=PyPDFLoader(pdf)
        pages=loader.load()
        resumetext= " ".join(page.page_content for page in pages)
        resumeinfo.append({"text": resumetext, "filename": filename})
    return resumeinfo
        

def main():
    st.header("Score Your resume based on Job Descriptions")
    st.subheader("Select best resume out of scores from the HR Consultant")
    resumefiles=[]
    filenames = []  
    for i in range(3):
        resume=st.file_uploader(f"Upload your Resumes {i+1} :")
        if resume is not None:
            st.success("Resume uploaded successfully..")
            filenames.append(resume.name) 
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as resume_file:
                resume_file.write(resume.getvalue())
                filepath=resume_file.name
            resumefiles.append(filepath)
            # for filepath in resumefiles:
            #     os.unlink(filepath)

    
    job_description=st.text_area("Now Past your targeted job description details") 
    if st.button("Start Compairing") :
        if len(resumefiles) < 3:
                st.warning("Please upload all 3 resumes first!")
                return
        if not job_description.strip():
                st.warning("Please enter a job description!")
                return
         # then proceed
        with st.spinner("Processing..."):
            results=[]
            chain=chain_creator()
            resume_texts=pdfProcessor(resumefiles, filenames)
            for resumetext in resume_texts:
                result=chain.invoke({"jd":job_description, "resume_info": resumetext})
                results.append(result)
            # results.sort(key=lambda x: x.score, reverse=True)

            for i, result in enumerate(results):
              result.filename = resume_texts[i]["filename"]
              results.sort(key=lambda x: x.score, reverse=True)

            for rank, result in enumerate(results):
                medal = ["🥇", "🥈", "🥉"][rank]
                with st.expander(f"{medal} {result.candidate_name} — Score: {result.score}/10 | 📄 {result.filename}"):
                    st.markdown(f"**📋 Summary:** {result.summary}")
                    st.markdown("**✅ Strengths:**")
                    for s in result.strength:
                        st.markdown(f"- {s}")
                    st.markdown("**❌ Weaknesses:**")
                    for w in result.weakness:
                        st.markdown(f"- {w}")

if __name__=="__main__":
    main()






