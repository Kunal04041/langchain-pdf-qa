import os
from groq import Groq
from dotenv import load_dotenv

# Load .env from the root
load_dotenv(os.path.join(os.path.dirname(__file__), "../../../../.env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class LLMService:
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment.")
        
        self.client = Groq(
            api_key=GROQ_API_KEY,
        )
        # Using a reliable free model on Groq
        self.primary_model = "llama-3.3-70b-versatile"

    def get_answer(self, context, question):
        prompt = f"""Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer concise and professional.

Context:
{context}

Question: {question}
Helpful Answer:"""

        try:
            completion = self.client.chat.completions.create(
                model=self.primary_model,
                messages=[
                    {"role": "system", "content": "You are a professional research assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3, # lower for more factual response
                max_tokens=1024,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            return f"Error connecting to Groq LLM: {str(e)}"
