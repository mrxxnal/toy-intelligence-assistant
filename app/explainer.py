from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def explain_match(query_image, row):
    """
    Generates human-style reasoning for why a toy matches another toy/image.
    """

    prompt = f"""
You are a helpful toy recommendation expert.

Explain WHY these two toys match in a natural, casual, human, easy-to-understand way and don't use hyphens.

Rules:
- Write like a real warm person explaining to a parent
- Avoid robotic phrases like "based on embeddings" or "vector similarity"
- Be specific and grounded
- Mention 2–3 clear reasons max
- Keep it short (3–5 sentences)

TOY 1 (query image):
- This is the user's uploaded toy image

TOY 2:
Name: {row['name']}
Category: {row['category']}
Description: {row['description']}

Now explain why they are similar.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.6
    )

    return response.choices[0].message.content