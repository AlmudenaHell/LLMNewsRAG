from openai import OpenAI
import os
import json

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_info(text, context):
    prompt = f"""
You are an information extraction system.

Given the following article and context, extract:
- company
- event
- category

Return ONLY a JSON object.

Article:
{text}

Context:
{context}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {"error": "Invalid JSON"}
