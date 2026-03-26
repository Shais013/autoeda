import json
import anthropic


def generate_insights(summary_dict: dict) -> str:
    """
    Sends the EDA summary to Claude API and gets back a plain English narrative.
    """
    client = anthropic.Anthropic()

    prompt = f"""
You are a senior data scientist reviewing a dataset. Below is an automated EDA summary in JSON format.

Write a clear, insightful narrative (5-8 sentences) for a non-technical audience covering:
1. What the dataset looks like overall (size, columns, types)
2. Any data quality issues (missing values, duplicates)
3. Interesting patterns, distributions, or correlations
4. Outliers or anomalies worth investigating
5. A 2-3 bullet point "Recommended Next Steps" section

Be specific, use actual numbers from the summary, and highlight the most important findings.

EDA Summary:
{json.dumps(summary_dict, indent=2, default=str)}
"""

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return message.content[0].text


def generate_column_insight(col_name: str, col_stats: dict) -> str:
    """Generates a short insight for a single column."""
    client = anthropic.Anthropic()

    prompt = f"""
In 2-3 sentences, describe what's notable about the column '{col_name}' based on these stats:
{json.dumps(col_stats, indent=2, default=str)}

Be concise and specific. Mention any red flags or interesting patterns.
"""

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=200,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return message.content[0].text
