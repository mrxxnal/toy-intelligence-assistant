from openai import OpenAI
import streamlit as st
import os

# ✅ SAFE KEY HANDLING (Streamlit + local)
def get_api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except:
        return os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=get_api_key())


def ask_toy_question(question, retrieved_toys):
    context = ""

    for _, row in retrieved_toys.iterrows():
        context += f"""
Toy Name: {row['name']}
Category: {row['category']}
Description: {row['description']}
Age: {row.get('age', 'Unknown')}
---
"""

    messages = [
        {
            "role": "system",
            "content": "You are an expert toy analyst. You recommend toys based only on provided data."
        },
        {
            "role": "user",
            "content": f"""
TOY DATA:
{context}

QUESTION:
{question}
"""
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message.content