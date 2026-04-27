from openai import OpenAI
import streamlit as st
import os

def get_client():
    try:
        return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except:
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

client = get_client()


def explain_match(query_image, toy_row):
    prompt = f"""
You are a toy recommendation AI.

Explain why this toy matches the user's uploaded image.

Be short and clear.

TOY:
Name: {toy_row['name']}
Category: {toy_row['category']}
Description: {toy_row['description']}
Age: {toy_row.get('age', 'Unknown')}

Rules:
- 2 to 4 sentences only
- Mention category or activity similarity
- Mention age suitability if relevant
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You explain recommendations clearly and simply."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content