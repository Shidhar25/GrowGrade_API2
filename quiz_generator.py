# quiz_generator.py
import os
import json
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Configure LLM (Gemini)
def get_llm():
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY is missing from .env file")
        
    return ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.2,
    )

llm = get_llm()


def generate_quiz_from_text(text, num_questions=10, difficulty="Medium"):
    """
    Force the LLM to return strict JSON array of MCQs.
    Each MCQ: {question, options[4], correct_answer (A/B/C/D)}
    """
    prompt = f"""
You are a professional quiz generator.

Given the CONTENT below, generate EXACTLY {num_questions} multiple-choice questions.
Difficulty Level: {difficulty}

Return ONLY a JSON array (no surrounding text, no Markdown):

[
  {{
    "question": "string",
    "options": ["A", "B", "C", "D"],
    "correct_answer": "A"
  }}
]

Rules:
- Must return valid JSON only.
- options must be exactly 4 strings each.
- correct_answer must be one of "A", "B", "C", "D".
- No explanations, no extra text, no code fences.

CONTENT:
{text}

Return ONLY the JSON array.
"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    raw = resp.content.strip()

    # Clean common wrappers like ```json ... ```
    cleaned = raw.replace("```json", "").replace("```", "").strip()

    # Try JSON parsing
    try:
        parsed = json.loads(cleaned)
    except Exception as e:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start:end+1])
            except Exception as e2:
                raise ValueError(
                    f"Failed to parse JSON from LLM output.\n"
                    f"Original error: {e}\nSecond: {e2}\nRaw:\n{raw}"
                )
        else:
            raise ValueError(
                f"Failed to parse JSON from LLM output.\nError: {e}\nRaw:\n{raw}"
            )

    return parsed