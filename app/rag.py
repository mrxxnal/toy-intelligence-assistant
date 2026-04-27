from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_toy_question(question, retrieved_toys):
    context = ""

    for _, row in retrieved_toys.iterrows():
        context += f"""
Toy Name: {row['name']}
Category: {row['category']}
Description: {row['description']}
---
"""

    prompt = f"""
You are a toy expert assistant.

Use the following toy data to answer the question.

TOY DATA:
{context}

QUESTION:
{question}

Rules:
- Be helpful and safe
- If unsure, say so
- Base answer only on given toys
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content