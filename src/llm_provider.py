# from langchain_ollama import OllamaLLM, OllamaEmbeddings
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from src.config import (
#     OLLAMA_BASE_URL, DEFAULT_MODEL, EMBEDDING_MODEL, SIMILARITY_MODEL,
#     OPENAI_API_KEY, OPENAI_MODEL, OPENAI_EMBEDDING_MODEL, OPENAI_SIMILARITY_MODEL,
#     USE_OPENAI, base_url
# )

# class LLMProvider:
#     def __init__(self):
#         if USE_OPENAI:
#             # 使用 ChatOpenAI 并启用 JSON 模式
#             self.llm = ChatOpenAI(
#                 model=OPENAI_MODEL,
#                 api_key=OPENAI_API_KEY,
#                 base_url=base_url,
#                 temperature=0,
#             ).bind(response_format={"type": "json_object"})
#             self.embedding_model = OpenAIEmbeddings(
#                 model=OPENAI_EMBEDDING_MODEL,
#                 api_key=OPENAI_API_KEY,
#                 base_url=base_url
#             )
#             self.similarity_model = ChatOpenAI(
#                 model=OPENAI_SIMILARITY_MODEL,
#                 api_key=OPENAI_API_KEY,
#                 base_url=base_url,
#                 temperature=0,
#             ).bind(response_format={"type": "json_object"})
#         else:
#             # 使用 Ollama 模型
#             self.llm = OllamaLLM(
#                 model=DEFAULT_MODEL,
#                 base_url=OLLAMA_BASE_URL,
#                 format='json',
#                 temperature=0
#             )
#             self.embedding_model = OllamaEmbeddings(
#                 model=EMBEDDING_MODEL,
#                 base_url=OLLAMA_BASE_URL
#             )
#             self.similarity_model = OllamaLLM(
#                 model=SIMILARITY_MODEL,
#                 base_url=OLLAMA_BASE_URL,
#                 format='json',
#                 temperature=0
#             )

#     def get_llm(self):
#         return self.llm

#     def get_embedding_model(self):
#         return self.embedding_model

#     def get_similarity_model(self):

#         return self.similarity_model


import os
import re
import time
import uuid
from pathlib import Path
from fastapi import FastAPI, APIRouter
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor

from app.retrieval.query import retrieve, answer_from_context
from app.agents.router import classify_dept, detect_action
try:
    from app.kag.traverse_neo4j import traverse as graph_traverse
except Exception:
    from app.kag.text2graph import traverse as graph_traverse

from app.agents.agents import REGISTRY as AGENTS
from app.tools.hr import router as hr_tools
from app.tools.it import router as it_tools
from app.tools.finance import router as fin_tools
from app.memory.store import add_turn, get_history

# Optional: ngrok integration for Colab environment
RUNNING_IN_COLAB = "COLAB_GPU" in os.environ

if RUNNING_IN_COLAB:
    from pyngrok import ngrok
    import nest_asyncio
    nest_asyncio.apply()
    public_url = ngrok.connect(addr=8000, bind_tls=True).public_url
    print(f" * ngrok tunnel running at {public_url}")

app = FastAPI()
app.mount('/assets', StaticFiles(directory='app/web/assets'), name='assets')

tools = APIRouter(prefix='/tools')
tools.include_router(hr_tools, prefix='/hr', tags=['tools-hr'])
tools.include_router(it_tools, prefix='/it', tags=['tools-it'])
tools.include_router(fin_tools, prefix='/finance', tags=['tools-finance'])
app.include_router(tools)

class ChatIn(BaseModel):
    text: str
    dept: str | None = None  # "hr" | "it" | "finance"
    conversation_id: str | None = None

def _is_small_talk(text: str) -> bool:
    t = text.strip().lower()
    patterns = [
        r"^(hi|hello|hey)\b",
        r"how are you\b",
        r"^(good\s+)?(morning|afternoon|evening)\b",
        r"\bthank(s| you)\b",
        r"^(bye|goodbye)\b",
        r"what's up\b",
    ]
    return any(re.search(p, t) for p in patterns)

def _small_talk_reply(text: str) -> str:
    t = text.strip().lower()
    if re.search(r"how are you", t):
        return "I'm doing well and ready to help. What do you need from HR, IT, or Finance today?"
    if re.search(r"thank", t):
        return "You're welcome! If you need anything else across HR, IT, or Finance, just ask."
    if re.search(r"(morning|afternoon|evening)", t):
        return "Hello! How can I assist you with HR, IT, or Finance today?"
    if re.search(r"(bye|goodbye)", t):
        return "Goodbye! Have a great day."
    return "Hello! How can I assist you with HR, IT, or Finance today?"

@app.post('/chat')
def chat_endpoint(payload: ChatIn):
    trace_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    conv_id = payload.conversation_id or trace_id
    add_turn(conv_id, 'user', payload.text)

    if _is_small_talk(payload.text):
        ans = _small_talk_reply(payload.text)
        add_turn(conv_id, 'assistant', ans)
        t4 = time.perf_counter()
        return {
            'trace_id': trace_id,
            'conversation_id': conv_id,
            'dept': 'general',
            'action': 'small_talk',
            'answer': ans,
            'timings': {'total_ms': round((t4 - t0) * 1000, 2)}
        }

    dept = (payload.dept or classify_dept(payload.text)).lower()
    t1 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_hints = ex.submit(graph_traverse, payload.text)
        f_ctx = ex.submit(retrieve, dept, payload.text)
        hints = f_hints.result()
        t2 = time.perf_counter()
        ctx = f_ctx.result()
        t3 = time.perf_counter()
    agent = AGENTS.get(dept)
    
    history = get_history(conv_id)
    if history:
        hist_text = "\n".join([f"{m['role']}: {m['content']}" for m in history[-6:]])
        ctx = [("Conversation history:\n" + hist_text, 1.0)] + ctx

    result = agent.handle(payload.text, ctx, hints) if agent else {'dept': dept, 'answer': "No agent found"}
    add_turn(conv_id, 'assistant', result.get('answer', ''))

    t4 = time.perf_counter()
    result.update({
        'trace_id': trace_id,
        'conversation_id': conv_id,
        'timings': {
            'total_ms': round((t4 - t0) * 1000, 2),
            'kag_ms': round((t2 - t1) * 1000, 2),
            'retrieve_ms': round((t3 - t2) * 1000, 2),
            'generate_ms': round((t4 - t3) * 1000, 2),
        }
    })
    return result

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.get('/', response_class=HTMLResponse)
def index():
    return Path('app/web/index.html').read_text()

from pydantic import BaseModel as _BM
class FeedbackIn(_BM):
    trace_id: str
    text: str
    dept: str | None = None
    answer: str | None = None
    rating: str
    comment: str | None = None

@app.post('/feedback')
def feedback(inb: FeedbackIn):
    import json, time
    rec = inb.model_dump() | {'ts': int(time.time())}
    with open('feedback.jsonl', 'a') as f:
        f.write(json.dumps(rec) + '\n')
    return {'ok': True}
