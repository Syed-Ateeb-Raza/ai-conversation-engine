import re
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional, Dict

from common import get_model, remove_emojis, add_cors, verify_token

app = FastAPI()
add_cors(app)

def clean_text(text, llm_model):
    if llm_model == "qwen":
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
        text = re.sub(r"#\w+", "", text)
        text = remove_emojis(text)
        text = text.strip()

    elif llm_model == "llama":
        text = re.sub(r"^[^:]{1,100}:\s*", "", text)
        text = re.sub(r"[\*\#]", "", text)
        text = re.sub(r",", "", text)
        text = re.sub(r"#\w+", "", text)
        text = remove_emojis(text)
        text = re.sub(r"\s+", " ", text).strip()

    elif llm_model == "claude":
        text = re.sub(r"^Assistant:\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^[^:]{1,100}:\s*", "", text)
        text = re.sub(r"\*", "", text)
        text = re.sub(r"[-–—]", " ", text)
        text = re.sub(r"#\w+", "", text)
        text = re.sub(r"'[^']*'\s*$", "", text)
        text = re.sub(r"\s+\.", ".", text)
        text = re.sub(r"This topic.*", "", text, flags=re.IGNORECASE | re.DOTALL)
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
        text = re.sub(r"#\w+", "", text)

    text = re.sub(r"[-–—]", " ", text)
    text = text.replace('"', '').replace('!', '.').strip()
    return text

detect_type_system = (
    "Analyze the text and Classify the following text strictly as either QUESTION or KEYWORD.\n"
    "If it is a question (ends with a question mark or is phrased like one), output only: QUESTION.\n"
    "Otherwise, output only: KEYWORD.\n"
    "Do not add explanations, punctuation, or any extra words."
)

detect_type_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", detect_type_system),
        ("user", "{text}")
    ]
)
# --- If it's a question, generate 3 similar questions using the SAME LLM ---
similar_questions_system = (
    "You are a helpful assistant that rewrites a given question into 3 distinct, natural, human-like questions.\n"
    "Keep the meaning close but vary the phrasing. Use the specified language.\n"
    "Adapt the rewrites according to tone, target audience, user preferences, Sentiment bias.\n"
    "Return ONLY the 3 questions, numbered 1-3, one per line. No extra text."
)

similar_questions_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", similar_questions_system),
        (
            "user",
            "Original question: {question}\n"
            "Language: {language}\n"
            "Character limit per question: {char_len}\n"
            "Tone: {tone}\n"
            "Target Audience: {target_audience}\n"
            "User Preference: {user_preference}\n"
            "Sentiment Bias: {sentiment_bias}\n"
            "Generate exactly 3 variations, numbered 1-3."
        ),
    ]
)

llama_system_template = (
    "You're someone who casually comes up with blog and forum topics that feel personal, real, and totally human. Forget sounding like a machine—your ideas should have personality, quirks, maybe even a touch of humor or sarcasm when it fits. Think like you're chatting with a friend or leaving a thoughtful (but chill) comment online. Authenticity is more important than perfection. Don't worry about being too polished—use natural language. Keep SEO in mind but don't force keywords—just make it feel like it flows in naturally, like how someone would actually write."
)
llama_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", llama_system_template),
        ("user", 
        "Generate an SEO-friendly discussion topic of {char_len} characters. "
        "Use the keyword '{topic}' naturally in a '{tone}' tone. Make it feel human and relatable for '{target_audience}', reflecting '{user_preference}'. "
        "Maintain a '{sentiment_bias}' sentiment and write in '{language}'. Use real-world analogies, rhetorical questions, contractions, and a casual style. "
        "Avoid generic intros like 'Hey gamers!'—go for specificity.")
    ]
)

deepseek_system_template = (
    "You are a creative and conversational discussion topic generator for forums and blogs. Your goal is to craft engaging, search-optimized topics that feel natural, relatable, and written by a real person. Use clear, everyday language and focus on connection over perfection. When appropriate, include light humor, rhetorical questions, or trending phrases that resonate with the intended audience. Avoid sounding robotic or overly polished—prioritize authenticity and flow while naturally including relevant keywords for SEO."
)

deepseek_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", deepseek_system_template),
        ("user", "Generate an SEO-friendly discussion topic of {char_len} characters using the keyword '{topic}', come up with a natural-sounding in a '{tone}' tone. Make it engaging and relatable for '{target_audience}', and reflect '{user_preference}' in style and phrasing. Keep the overall vibe '{sentiment_bias}', use '{language}'. The topic should sound like something a real person would post on a forum or blog—not stiff or robotic. Feel free to be a little playful or curious if it fits.")
    ]
)

qwen_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", llama_system_template),
        ("user", "Generate an SEO-friendly discussion topic of {char_len} characters using the keyword '{topic}', come up with a natural-sounding, SEO-friendly discussion topic in a '{tone}' tone. Make it engaging and relatable for '{target_audience}', and reflect '{user_preference}' in style and phrasing. Keep the overall vibe '{sentiment_bias}', use '{language}'. The topic should sound like something a real person would post on a forum or blog—not stiff or robotic. Feel free to be a little playful or curious if it fits.")
    ]
)

open_ai_system_template = (
    "You are an SEO-optimized discussion topic generator for forums and blogs, but your writing should feel completely natural and human-like."
    "Your task is to generate compelling, keyword-rich discussion topics that rank well in search engines while sounding authentic and conversational."
    "You will be given a topic keyword, tone of voice, word length, target audience, user preference, sentiment bias, and language."
    "Ensure that the topic is engaging and easy to understand by using natural phrasing, contractions, and varied sentence structures."
    "Avoid complex jargon and excessive neutrality—introduce subtle opinions, relatability, and occasional rhetorical questions to make it sound human."
    "Avoid generic introductory phrases such as 'Music Lovers Unite:', 'Calling All Fans:', or 'Attention Gamers!'."
    "Use specific references or real-world analogies to make the discussion topic feel more authentic and engaging."
)

open_ai_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", open_ai_system_template),
        ("user", "Generate an SEO-friendly discussion topic of {char_len} characters using the keyword '{topic}' in a '{tone}' tone of voice. "
                 "Make it feel natural and engaging for '{target_audience}', ensuring it aligns with '{user_preference}'. "
                 "Maintain a '{sentiment_bias}' sentiment and write in '{language}'. "
                 "Use contractions, rhetorical questions, and real-world analogies to make it sound human. "
                 "Avoid generic introductions like 'Attention all gamers!' and instead use specific, relatable scenarios. "
                 )
    ]
)

claude_system_template = (
    "You are an SEO-optimized discussion topic generator for forums and blogs, but your writing should feel completely natural and human-like."
    "Your task is to generate compelling, keyword-rich discussion topics that rank well in search engines while sounding authentic and conversational."
    "You will be given a topic keyword, tone of voice, word length, target audience, user preference, sentiment bias, and language."
    "Ensure that the topic is engaging and easy to understand by using natural phrasing, contractions, and varied sentence structures."
    "Avoid complex jargon and excessive neutrality—introduce subtle opinions, relatability, and occasional rhetorical questions to make it sound human."
    "Avoid generic introductory phrases such as 'Music Lovers Unite:', 'Calling All Fans:', or 'Attention Gamers!'."
    "Use specific references or real-world analogies to make the discussion topic feel more authentic and engaging."
)

claude_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", claude_system_template),
        ("user", "Generate an SEO-friendly discussion topic of {char_len} characters using the keyword '{topic}' in a '{tone}' tone of voice. "
                 "Make it feel natural and engaging for '{target_audience}', ensuring it aligns with '{user_preference}'. "
                 "Maintain a '{sentiment_bias}' sentiment and write in '{language}'. "
                 "Use contractions, rhetorical questions, and real-world analogies to make it sound human. "
                 "Avoid generic introductions like 'Attention all gamers!' and instead use specific, relatable scenarios. "
                 )
    ]
)

gemini_system_template = (
    "You are a creative and conversational discussion topic generator for forums and blogs. Your goal is to craft engaging, search-optimized topics that feel natural, relatable, and written by a real person. Use clear, everyday language and focus on connection over perfection. When appropriate, include light humor, rhetorical questions, or trending phrases that resonate with the intended audience. Avoid sounding robotic or overly polished—prioritize authenticity and flow while naturally including relevant keywords for SEO."
)

gemini_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", gemini_system_template),
        ("user", "Generate an SEO-friendly discussion topic of {char_len} characters using the keyword '{topic}', come up with a natural-sounding in a '{tone}' tone. Make it engaging and relatable for '{target_audience}', and reflect '{user_preference}' in style and phrasing. Keep the overall vibe '{sentiment_bias}', use '{language}'. The topic should sound like something a real person would post on a forum or blog—not stiff or robotic. Feel free to be a little playful or curious if it fits.")
    ]
)

simplify_system_template = (
    "Change the complex words into simple words in same language. "
    "Do not change the single quotes text. Do not include quotes. Only change the complex words that suites the sentence. "
    "It should look like human written. Start it as you are asking from human. "
    "Do not use any of irrelavent sentences such as 'I changed the following complex words to simpler ones' in the response. "
)

PROMPT_TEMPLATES = {
    "llama": llama_prompt_template,
    "qwen": qwen_prompt_template,
    "open-ai": open_ai_prompt_template,
    "claude": claude_prompt_template,
    "gemini": gemini_prompt_template,
    "deepseek": deepseek_prompt_template,
}

def simplify_text(text: str, char_len: Optional[int] = None):
    user_msg = "Change the text into simple words {text}."
    if char_len is not None:
        user_msg = "Change the text into simple words {text}. Characters length should not be increased than {char_len}"
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", simplify_system_template),
        ("user", user_msg)
    ])
    chain = prompt_template | get_model("llama")
    payload = {"text": text} if char_len is None else {"text": text, "char_len": char_len}
    return chain.invoke(payload).content

class TopicRequest(BaseModel):
    id: str
    model: str
    topic: str
    tone: str
    char_len: int
    target_audience: str
    user_preference: str
    sentiment_bias: str
    language: str

class TopicResponse(BaseModel):
    id: str
    generated_topic: Optional[str] = None
    generated_topic_array: Optional[Dict[str, str]] = None

@app.post("/generate-topic", response_model=TopicResponse, response_model_exclude_none=True, dependencies=[Depends(verify_token)])
async def generate_topic(request: TopicRequest):
    """
    Generate a discussion topic based on the given parameters.
    If model == request.model determines the input is a QUESTION, return 3 similar questions.
    If it's a KEYWORD, preserve existing logic unchanged.
    """
    try:
        # Initialize the selected model (this SAME model will be used for detection & generation)
        model = get_model(request.model)
        
        if request.model == "claude" and request.char_len > 190:
            request.char_len = 190

        # --- 1) Use SAME LLM to detect if input is QUESTION or KEYWORD ---
        detect_chain = detect_type_prompt | model
        type_raw = detect_chain.invoke({"text": request.topic}).content.strip()
        type_raw_cleaned = clean_text(type_raw, request.model)
        # Defensive parsing
        if "QUESTION" in type_raw_cleaned:
            is_question = True
        elif "KEYWORD" in type_raw_cleaned:
            is_question = False
        else:
            # fallback: simple heuristic
            is_question = request.topic.strip().endswith("?")

        if is_question:
            sim_chain = similar_questions_prompt | model
            sim_res = sim_chain.invoke({
                "question": request.topic,
                "language": request.language,
                "char_len": request.char_len,
                "tone": request.tone,
                "target_audience": request.target_audience,
                "user_preference": request.user_preference,
                "sentiment_bias": request.sentiment_bias,
            })

            cleaned_text = clean_text(sim_res.content, request.model)
            raw_text = cleaned_text.strip()

            simplified = simplify_text(raw_text)
            cleaned_text = clean_text(simplified, request.model)

            # Split by newlines or numbering
            lines = [q.strip("•-–—*012345. )(").strip()
                    for q in re.split(r"\n|(?<=\?)\s+", cleaned_text)
                    if q.strip()]

            # Keep only first 3 questions
            questions_dict = {f"{i+1}": line for i, line in enumerate(lines[:3])}
            return TopicResponse(id=request.id, generated_topic_array=questions_dict)

        # --- 3) Otherwise KEYWORD: keep your existing logic unchanged ---
        prompt_template = PROMPT_TEMPLATES.get(request.model, deepseek_prompt_template)

        chain = prompt_template | model
        result = chain.invoke({
            "topic": request.topic,
            "tone": request.tone,
            "char_len": request.char_len,
            "target_audience": request.target_audience,
            "user_preference": request.user_preference,
            "sentiment_bias": request.sentiment_bias,
            "language": request.language
        })
        generated_content = result.content
        cleaned_text = clean_text(generated_content, request.model)

        simplified = simplify_text(cleaned_text, request.char_len)
        cleaned_text = clean_text(simplified, request.model)
        return TopicResponse(id=request.id, generated_topic=cleaned_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=5000)