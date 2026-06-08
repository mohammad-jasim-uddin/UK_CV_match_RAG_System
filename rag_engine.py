from openai import OpenAI
from src.prompts import build_cv_match_prompt


class RAGEngine:
    """Retrieves relevant CV sections and asks an LLM to create a job-match report."""

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.client = OpenAI()

    def analyse_cv_match(self, job_description: str) -> str:
        retrieved_sections = self.vector_store.search(
            query=job_description,
            top_k=5,
        )

        prompt = build_cv_match_prompt(
            job_description=job_description,
            retrieved_cv_sections=retrieved_sections,
        )

        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert UK recruitment, ATS, and CV matching assistant.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content or "No response generated."
