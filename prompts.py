
def build_cv_match_prompt(job_description: str, retrieved_cv_sections: list[str]) -> str:
    context = "\n\n--- CV SECTION ---\n\n".join(retrieved_cv_sections)

    return f"""
You are a UK recruitment, ATS, and CV screening assistant.

Compare the candidate CV sections against the UK job description.

JOB DESCRIPTION:
{job_description}

RELEVANT CV SECTIONS RETRIEVED BY RAG:
{context}

Create a detailed CV-job match report with these sections:

1. Overall Match Score out of 100
2. Education Match
3. Skills Match
4. Experience Match
5. Project Match
6. Missing Skills or Weak Areas
7. Candidate Strengths
8. UK Job Market Suitability
9. ATS Keyword Suggestions
10. CV Improvement Suggestions
11. Final Recruitment Recommendation

Be specific, professional, practical, and UK-market focused.
"""
