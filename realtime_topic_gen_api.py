import os
import re
import json
import requests
import uvicorn
from typing import Dict, Any
from fastapi import FastAPI, Query, HTTPException

from common import get_model, add_cors

app = FastAPI()
add_cors(app)

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM text output or attempt to repair it."""
    match = re.search(r'\{[\s\S]*\}', text.strip())
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # --- Attempt to recover: parse numbered lines ---
    recovered = {}
    numbered_lines = re.findall(r'(\d+)\.\s*(.+)', text)
    for num, content in numbered_lines:
        recovered[num.strip()] = content.strip()

    if recovered:
        return recovered

    # fallback: single key if nothing valid
    return {"1": text.strip() or "No valid JSON or numbered text found."}

def parse_error_message(error: Exception) -> Dict[str, str]:
    """Convert common API/LLM error messages into user-friendly responses."""
    msg = str(error).lower()

    # Map common error codes or phrases
    if "429" in msg or "rate limit" in msg:
        return {"error": "Rate limit exceeded. Please wait and try again."}
    elif "401" in msg or "unauthorized" in msg:
        return {"error": "Unauthorized request. Check your API key or credentials."}
    elif "403" in msg or "forbidden" in msg:
        return {"error": "Access forbidden. You might not have permission to use this model."}
    elif "400" in msg or "bad request" in msg:
        return {"error": "Bad request. The input or parameters may be invalid."}
    elif "timeout" in msg or "timed out" in msg:
        return {"error": "Request timed out. Please try again later."}
    elif "500" in msg or "server error" in msg:
        return {"error": "Internal server error from the model API."}
    elif "quota" in msg or "limit" in msg:
        return {"error": "Quota limit reached for this API plan."}
    elif "model" in msg and "not found" in msg:
        return {"error": "The requested model could not be found or is unavailable."}
    elif "invalid" in msg:
        return {"error": "Invalid input or API parameter detected."}
    else:
        return {"error": "Unexpected error occurred."}

@app.get("/search")
def search(
    query: str = Query(..., title="Search Query"),
    model: str = Query("llama", title="Model")
):
    """
    Performs a Google Serper search, combines snippets,
    and calls the LLM once to generate refined discussion-style topics.
    Always returns clean numbered JSON.
    """
    try:
        model = model.lower()
        llm = get_model(model)

        # --- SERPER API Call ---
        search_url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
        response = requests.post(search_url, headers=headers, json={"q": query, "gl": "pk"}, timeout=15)
        response.raise_for_status()
        search_data = response.json()

        snippets = [r.get("snippet", "") for r in search_data.get("organic", [])]
        if not snippets:
            return {"results": {}}

        snippets_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(snippets)])

        # --- LLM Prompt ---
        system_prompt = (
            "You are an SEO-optimized discussion topic generator for forums and blogs.\n"
            "You will receive multiple snippets labeled with numbers.\n"
            "For each snippet, write a refined discussion version keeping the text mostly unchanged "
            "but adding a natural discussion question at the start and end.\n"
            "Ensure the refined output is no longer than 255 characters.\n"
            "Return only valid JSON (no markdown, no extra text) like this:\n"
            "{\n"
            '  \"1\": \"Teamwork helps us grow. What are your thoughts? Let’s discuss!\",\n'
            '  \"2\": \"Honesty builds trust. Do you agree?\"\n'
            "}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": snippets_text}
        ]

        llm_response = llm.invoke(messages)
        raw_output = getattr(llm_response, "content", str(llm_response)).strip()

        refined_results = extract_json_from_text(raw_output)
        return {"results": refined_results}

    except requests.exceptions.RequestException as e:
        parsed = parse_error_message(e)
        parsed["details"] = str(e)
        return parsed

    except Exception as e:
        parsed = parse_error_message(e)
        parsed["details"] = str(e)
        return parsed

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5003)
