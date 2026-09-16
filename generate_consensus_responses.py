import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from typing import Optional, List

from common import get_model, clean_generated_text, add_cors, verify_token, AVAILABLE_MODELS

app = FastAPI()
add_cors(app)

def build_simple_text_with_errors_prompt(error_types: List[str]) -> str:
    if not error_types:
        return (
            "You're someone who changes complex words into simple words. "
            "Do not change the text inside single quotes. Only simplify complex words where needed. "
            "It should look like a human wrote it. Avoid any irrelevant phrases like "
            "'I changed the following complex words to simpler ones' in your response."
        )

    error_instructions = {
        "spelling": "- Add spelling mistakes (e.g., 'definately' for 'definitely')",
        "grammar": "- Add grammar mistakes (e.g., missing articles or incorrect verb tense)",
        "punctuation": "- Add punctuation mistakes (e.g., missing periods or commas)"
    }

    allowed = set(error_types)
    forbidden = {"spelling", "grammar", "punctuation"} - allowed

    selected_instructions = "\n".join([error_instructions[e] for e in allowed])
    forbidden_instructions = "\n".join([f"- DO NOT add {e} mistakes." for e in forbidden])

    return (
        f"You're someone who changes complex words into simple words and adds only the following human-like mistakes:\n"
        f"{selected_instructions}\n"
        f"{forbidden_instructions}\n\n"
        "Do not change the text inside single quotes. Only simplify complex words where needed. "
        "Make sure the response looks like it was written casually by a human. "
        "Do not include any extra commentary or meta-explanation."
    )

general_system_template = (
    "You are a response generator for forums. Your role is to create responses that feel conversational, natural and friendly. "
    "Ensure the tone is relaxed and approachable, just like you're chatting with a friend. "
    "If appropriate, you may encourage engagement, but do not force it."
    "Use simple and everyday language, and avoid sounding too formal or robotic. "
    "Your goal is to keep the conversation light and fun while making it clear, engaging, and human-like."
)

general_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", general_system_template),
        (
            "user",
            "Conversation so far: {history}\n\nNow respond to: '{text}' "
            "considering user preferences '{user_preference}', "
            "in a '{tone}' tone of voice, with a '{sentiment_bias}' sentiment bias. "
            "Keep the response natural, friendly, and conversational, like you're talking to a friend. "
            "Provide a direct response without asking follow-up questions or prompting further discussion. "
            "Keep the response to {words_len} words."
        ),
    ]
)

open_ai_system_template = (
    "You're a friendly, casual forum participant. Your responses should sound natural, like a real person chatting. "
    "Keep it relaxed, engaging, and slightly imperfect—like how people actually talk online. "
    "Avoid sounding overly polished, robotic, or like a corporate AI. "
    "Use contractions, mix in informal phrasing, and don't be afraid to add personality. "
    "Feel free to throw in humor, light sarcasm, or relatable commentary where it makes sense. "
    "Your goal is to make responses feel effortless and authentic, like a real conversation."
)

open_ai_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", open_ai_system_template),
        (
            "user",
            "Alright, here's what's been said so far: {history}\n\n"
            "Now, respond to: '{text}' in a way that aligns with '{user_preference}', "
            "with a '{tone}' tone and a '{sentiment_bias}' sentiment bias. "
            "Keep it casual, like you're just replying to a message from a friend. "
            "Use natural phrasing—avoid sounding too polished or robotic. "
            "Don't force engagement, but if it fits, you can add humor, sarcasm, or a little personality. "
            "Stick to {words_len} words."
        ),
    ]
)

avoid_reasoning_system_prompt = (
    "You are a strict output validator.\n"
    "Your job is to decide if the given text is REASONING or a FINAL VALID ANSWER.\n\n"
    "REASONING includes:\n"
    "- Explanations of thinking\n"
    "- Mentions of 'I think', 'let me', 'the user', 'this task'\n"
    "- Analysis, justification, or meta commentary\n\n"
    "VALID OUTPUT includes:\n"
    "- Direct answer to the topic\n"
    "- No explanation or meta text\n"
    "- Sounds like a final forum reply\n\n"
    "Respond with ONLY one word:\n"
    "- reasoning\n"
    "- valid"
)

avoid_reasoning_prompt = ChatPromptTemplate.from_messages([
    ("system", avoid_reasoning_system_prompt),
    ("user", "{text}")
])

class TextRequest(BaseModel):
    id: str
    text: str
    tone: str
    words_len: int
    user_preference: str
    sentiment_bias: str
    history: list
    intentional_errors: Optional[bool] = False
    error_types: Optional[List[str]] = []
    models: Optional[List[str]] = None

class TextResponse(BaseModel):
    id: str
    generated_response: str

def generate_consensus_response(responses: List[str], words_length: int, fallback_model_key: str = "llama") -> str:
    formatted = "\n".join(f"{i+1}. {r}" for i, r in enumerate(responses))
    prompt = f"""
    Below are {len(responses)} responses.
    Your task is to write a single final answer that ONLY includes points mentioned in ALL responses.
    Do NOT explain, analyze, or describe anything.
    Do NOT add extra words or reasoning.

    Responses:
    {formatted}

    Return ONE short answer in {words_length} words.
    ONLY return the final answer. Nothing else.
    """
    try:
        model = get_model(fallback_model_key)
        response = model.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        raise RuntimeError(f"Consensus model '{fallback_model_key}' failed: {e}")

@app.post("/generate-consensus-response", response_model=TextResponse, dependencies=[Depends(verify_token)])
async def generate_response(request: TextRequest):
    results = {}
    model_responses = []
    successful_models = []

    model_keys = request.models if request.models else list(AVAILABLE_MODELS.keys())

    # Validate user-provided models
    invalid_models = [m for m in model_keys if m not in AVAILABLE_MODELS]
    if invalid_models:
        raise HTTPException(status_code=400, detail=f"Invalid model(s): {invalid_models}. Choose from {list(AVAILABLE_MODELS.keys())}")

    for model_key in model_keys:
        try:
            model = get_model(model_key)
            prompt_template = open_ai_prompt_template if model_key == "open-ai" else general_prompt_template
            chain = prompt_template | model

            result = chain.invoke({
                "history": "\n".join(request.history) if request.history else "",
                "text": request.text,
                "tone": request.tone,
                "words_len": request.words_len,
                "user_preference": request.user_preference,
                "sentiment_bias": request.sentiment_bias
            })

            cleaned = clean_generated_text(result.content, model_key)
            model_responses.append(cleaned)
            successful_models.append(model_key)

        except Exception as e:
            results[model_key] = f"Error: {str(e)}"
            continue

    if not model_responses:
        raise HTTPException(status_code=500, detail="No valid responses to consolidate.")

    # Pick the first working model as consensus generator
    fallback_model = successful_models[0]
    consensus_response = generate_consensus_response(model_responses, request.words_len, fallback_model)

    # Apply simplification or error injection using llama
    system_prompt = build_simple_text_with_errors_prompt(request.error_types if request.intentional_errors else [])
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Change the text into simple words\n\n{text}")
    ])

    simplify_chain = prompt_template | get_model("llama")
    classifier_chain = avoid_reasoning_prompt | get_model("llama")

    MAX_ATTEMPTS = 5
    attempt = 0

    # -------- First generation --------
    result = simplify_chain.invoke({"text": consensus_response})
    simplified = clean_generated_text(result.content, "llama")

    classification = classifier_chain.invoke({
        "text": simplified
    }).content.strip().lower()

    # -------- Retry ONLY if reasoning --------
    while classification == "reasoning" and attempt < MAX_ATTEMPTS:
        attempt += 1

        # regenerate consensus with SAME params
        consensus_response = generate_consensus_response(
            model_responses,
            request.words_len,
            fallback_model
        )

        # simplify again
        result = simplify_chain.invoke({"text": consensus_response})
        simplified = clean_generated_text(result.content, "llama")

        # re-classify
        classification = classifier_chain.invoke({
            "text": simplified
        }).content.strip().lower()

    return TextResponse(id=request.id, generated_response=simplified)

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=5004)
