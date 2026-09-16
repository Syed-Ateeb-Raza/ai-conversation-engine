import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate

from common import get_model, clean_text, add_cors, verify_token

app = FastAPI()
add_cors(app)

comment_filter_system = (
    "You are a strict binary classifier for social media comments.\n"
    "\n"
    "Respond with ONLY one word: WORTHY or NOT_WORTHY.\n"
    "Do not add explanations or extra words.\n"
    "\n"
    "WORTHY =\n"
    "- Genuine question.\n"
    "- Feedback or opinion that shows thought.\n"
    "- Interest in the product/brand/topic.\n"
    "- Asking for help, guidance, or clarification.\n"
    "\n"
    "NOT_WORTHY =\n"
    "- Comments with only emojis.\n"
    "- Single-word generic praise (e.g., 'nice', 'cool').\n"
    "- Spam or self-promotion (e.g., 'follow me', 'check profile').\n"
    "- Toxic, hateful, or offensive content.\n"
    "- Irrelevant or meaningless replies.\n"
)

comment_filter_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", comment_filter_system),
        ("user", "{comment}")
    ]
)

class CommentRequest(BaseModel):
    id: str
    model: str
    comment: str

class CommentResponse(BaseModel):
    id: str
    decision: str

@app.post("/check-worth", response_model=CommentResponse, dependencies=[Depends(verify_token)])
async def check_worth(request: CommentRequest):
    """
    Filter social media comments to decide if bot should reply.
    Returns WORTHY or NOT_WORTHY.
    """
    try:
        temperature = 0.6 if request.model in ("deepseek", "claude", "gemini") else 0
        model = get_model(request.model, temperature=temperature)

        # Run LLM classification
        filter_chain = comment_filter_prompt | model
        decision_raw = filter_chain.invoke({"comment": request.comment}).content.strip()
        # Clean + enforce strict output
        decision = clean_text(decision_raw, request.model)

        return CommentResponse(id=request.id, decision=decision)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=5005)
