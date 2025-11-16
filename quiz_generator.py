# quiz_generator.py
import os
import json
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# FIX: Ensure OpenAI API key env variable is set (ChatOpenAI requires this)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")  # IMPORTANT
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"


# Configure LLM (OpenRouter)
def get_llm():
    return ChatOpenAI(
        model="google/gemma-2-27b-it",
        api_key=os.getenv("OPENROUTER_API_KEY"),     # REQUIRED
        base_url="https://openrouter.ai/api/v1",     # REQUIRED
        temperature=0.2,
    )

llm = get_llm()


def generate_quiz_from_text(text, num_questions=10):
    """
    Force the LLM to return strict JSON array of MCQs.
    Each MCQ: {question, options[4], correct_answer (A/B/C/D)}
    """
    prompt = f"""
You are a professional quiz generator.

Given the CONTENT below, generate EXACTLY {num_questions} multiple-choice questions.
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
