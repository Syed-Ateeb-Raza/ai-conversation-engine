# AI Conversation Engine

FastAPI microservices that power AI-driven forum/community engagement: generating natural-sounding responses and discussion topics across multiple LLM providers, building multi-model consensus replies, and deciding when a bot should reply or stop replying.

## Services

| Service | Port | Endpoint | Purpose |
|---|---|---|---|
| `generate_response_api.py` | 5001 | `POST /generate-response` | Generates a single conversational reply from a chosen LLM, with optional simplification and human-like typo injection. |
| `generate_consensus_responses.py` | 5004 | `POST /generate-consensus-response` | Queries several LLMs for the same prompt and distills one consensus reply containing only the points every model agreed on. |
| `generate_topic_api.py` | 5000 | `POST /generate-topic` | Generates SEO-friendly discussion topics from a keyword, or rewrites a question into 3 natural variations. |
| `realtime_topic_gen_api.py` | 5003 | `GET /search` | Pulls live Google search snippets (via Serper) and turns them into discussion-ready topics. |
| `post_worth_checker_api.py` | 5005 | `POST /check-worth` | Classifies whether a comment is worth an automated reply (`WORTHY` / `NOT_WORTHY`). |
| `conversation_continuation_evaluator_api.py` | 5002 | `POST /generate-continuation-response` | Decides whether a conversation thread should continue or end. |

All services share model selection, LLM-output cleanup, CORS, and bearer-token auth logic from `common.py`.

## Supported models

`llama` (Groq), `qwen` (Groq), `deepseek`, `open-ai`, `claude`, `gemini`.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys and STATIC_TOKEN
python generate_response_api.py   # run any service the same way
```

Every endpoint requires an `Authorization: Bearer <STATIC_TOKEN>` header.

## GitHub repo description

> FastAPI microservices for AI-powered forum engagement — multi-LLM response and topic generation, consensus replies, and reply-worthiness/continuation classifiers.
