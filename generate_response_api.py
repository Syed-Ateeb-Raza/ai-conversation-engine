import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
from langchain_core.prompts import ChatPromptTemplate

from common import get_model, clean_generated_text, add_cors, verify_token

app = FastAPI()
add_cors(app)

def build_simple_text_with_errors_prompt(error_types: List[str]) -> str:
    if not error_types:
        return (
            "Simplify the text using everyday language. Do not alter text enclosed in single quotes. "
            "Avoid explanations your changes or extra commentary."
        )
    error_instructions = {
        "spelling": "- Add spelling mistakes (e.g., 'definately' for 'definitely')",
        "grammar": "- Add grammar issues (e.g., missing articles, subject-verb errors)",
        "punctuation": "- Add punctuation mistakes (e.g., missing or excessive punctuation)"
    }
    selected = "\n".join([error_instructions[e] for e in error_types if e in error_instructions])
    return (
        f"Rewrite the following text by simplifying it and adding these human-like mistakes:\n"
        f"{selected}\n\n"
        "Do not change anything inside single quotes. Do not explain your changes. "
        "Make the response sound naturally written by a human."
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
            "Keep the response to {words_len} words and provide the output in {language}."
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
            "Stick to {words_len} words and write in {language}."
        ),
    ]
)

class TextRequest(BaseModel):
    id: str
    model: str
    text: str
    tone: str
    words_len: int
    user_preference: str
    sentiment_bias: str
    language: str
    history: list
    intentional_errors: Optional[bool] = False
    error_types: Optional[List[str]] = []

class TextResponse(BaseModel):
    id: str
    generated_response: str

@app.post("/generate-response", response_model=TextResponse, dependencies=[Depends(verify_token)])
async def generate_response(request: TextRequest):
    try:
        model = get_model(request.model)

        # Choose prompt
        prompt_template = open_ai_prompt_template if request.model == "open-ai" else general_prompt_template

        chain = prompt_template | model
        result = chain.invoke({
            "history": "\n".join(request.history) if request.history else "",
            "text": request.text,
            "tone": request.tone,
            "words_len": request.words_len,
            "user_preference": request.user_preference,
            "sentiment_bias": request.sentiment_bias,
            "language": request.language
        })

        cleaned_text = clean_generated_text(result.content, request.model)

        # Apply optional simplification + error insertion
        transformer_model = get_model("llama")
        system_prompt = build_simple_text_with_errors_prompt(request.error_types if request.intentional_errors else [])
        transform_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Rewrite the following text:\n\n{text}")
        ])
        transform_chain = transform_prompt | transformer_model
        result = transform_chain.invoke({"text": cleaned_text})
        final_text = result.content.strip()

        return TextResponse(id=request.id, generated_response=final_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=5001)
