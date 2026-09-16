import os
import re
from typing import Optional

from dotenv import load_dotenv
from fastapi import Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

load_dotenv()

AVAILABLE_MODELS = {
    "deepseek": "deepseek-chat",
    "llama": "llama-3.3-70b-versatile",
    "qwen": "qwen/qwen3-32b",
    "open-ai": "gpt-4o",
    "claude": "claude-3-5-haiku-20241022",
    "gemini": "gemini-2.5-pro",
}

_MODEL_FACTORIES = {
    "open-ai": lambda t: ChatOpenAI(temperature=t, api_key=os.getenv("OPENAI_API_KEY"), model=AVAILABLE_MODELS["open-ai"]),
    "deepseek": lambda t: ChatDeepSeek(temperature=t, api_key=os.getenv("DEEPSEEK_API_KEY"), model=AVAILABLE_MODELS["deepseek"]),
    "claude": lambda t: ChatAnthropic(temperature=t, api_key=os.getenv("CLAUDE_API_KEY"), model=AVAILABLE_MODELS["claude"]),
    "gemini": lambda t: ChatGoogleGenerativeAI(temperature=t, api_key=os.getenv("GEMINI_API_KEY"), model=AVAILABLE_MODELS["gemini"]),
}


def get_model(model: str, temperature: float = 0.6):
    if model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model name. Choose from {list(AVAILABLE_MODELS.keys())}")
    factory = _MODEL_FACTORIES.get(model)
    if factory:
        return factory(temperature)
    return ChatGroq(temperature=temperature, groq_api_key=os.getenv("GROQ_API_KEY"), model_name=AVAILABLE_MODELS[model])


_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


def remove_emojis(text: str) -> str:
    return _EMOJI_PATTERN.sub("", text)


def clean_text(text: str, model: str) -> str:
    """Per-model output cleanup shared by response/consensus/worth-checker endpoints."""
    text = text.strip()
    if model == "qwen":
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        text = re.sub(r"(?i)(this topic|this discussion|this question|fosters|invites|encourages|engaging discussions).*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"[\[\(].*?[\]\)]", "", text)
        text = re.sub(r"^[^:\n]*:\s*", "", text)
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"\(.*?\)", "", text)
        text = re.sub(r"\*", "", text)
        text = re.sub(r"#.*", "", text)
        text = re.sub(r"'", "", text)
        text = re.sub(r"Word count:\s*\d+", "", text)
        text = re.sub(r"Character count:\s*\d+", "", text)
        text = re.sub(r"^Why it works:.*", "", text)
        text = re.sub(r",", "", text)
        text = re.sub(r"\n{2,}", "\n", text)
        text = remove_emojis(text)
    elif model == "llama":
        text = re.sub(r"^[^:]{1,100}:\s*", "", text)
        text = re.sub(r"[\*\#]", "", text)
        text = re.sub(r",", "", text)
        text = remove_emojis(text)
        text = re.sub(r"\s+", " ", text).strip()
    elif model == "claude":
        text = re.sub(r"^Assistant:\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^[^:]{1,100}:\s*", "", text)
        text = re.sub(r"\*", "", text)
        text = re.sub(r"[-–—]", " ", text)
        text = re.sub(r"#\w+", "", text)
        text = re.sub(r"'[^']*'\s*$", "", text)
        text = re.sub(r"\s+\.", ".", text)
        text = remove_emojis(text)
        text = re.sub(r"\s+", " ", text).strip()
    else:
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        text = re.sub(r"^[^:]{1,100}:\s*", "", text)
        text = re.sub(r"\(.*?\?.*?\)", "", text)
        text = re.sub(r"\*", "", text)
        text = re.sub(r"[-–—]", "", text)
        text = re.sub(r"\(Translation.*?\)", "", text)
        text = re.sub(r"keywords:\s*\d+", "", text)
        text = re.sub(r",", "", text)

    text = re.sub(r"[-–—]", " ", text)
    return text


def clean_generated_text(text: str, model: str) -> str:
    text = clean_text(text, model)
    return text.replace('"', '').replace('!', '.').replace(' - ', ' because ').strip()


def add_cors(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


STATIC_TOKEN = os.getenv("STATIC_TOKEN", "default_static_token")


async def verify_token(authorization: Optional[str] = Header(None)):
    if authorization is None:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    if authorization.strip() != f"Bearer {STATIC_TOKEN}":
        raise HTTPException(status_code=403, detail="Invalid or missing token")
    return True
