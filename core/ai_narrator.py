import json
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError(
        "GEMINI_API_KEY is not set. "
        "Add it to a .env file in the project root or to Streamlit Secrets."
    )

client = genai.Client(api_key=api_key)


def generate_insights(summary_dict: dict) -> str:
    prompt = f"""
You are a senior data scientist reviewing a dataset. Below is an automated EDA summary in JSON format.

Write a clear, insightful narrative (5-8 sentences) for a non-technical audience covering:
1. What the dataset looks like overall (size, columns, types)
2. Any data quality issues (missing values, duplicates)
3. Interesting patterns, distributions, or correlations
4. Outliers or anomalies worth investigating
5. A 2-3 bullet point Recommended Next Steps section

Be specific, use actual numbers from the summary, and highlight the most important findings.

EDA Summary:
{json.dumps(summary_dict, indent=2, default=str)}
"""

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt,
    )
    return response.text


def generate_column_insight(col_name: str, col_stats: dict) -> str:
    prompt = f"""
In 2-3 sentences, describe what is notable about the column '{col_name}' based on these stats:
{json.dumps(col_stats, indent=2, default=str)}

Be concise and specific. Mention any red flags or interesting patterns.
"""

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt,
    )
    return response.text
