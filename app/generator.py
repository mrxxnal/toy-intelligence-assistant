from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Create client with key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_toy_insights(name, description, category):
    prompt = f"""
You are a toy expert and product marketer.

Given the following toy:

Name: {name}
Category: {category}
Description: {description}

Generate:
1. A clear, engaging product description
2. Recommended age range
3. A short marketing pitch (2-3 sentences)

Keep it concise and realistic.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content