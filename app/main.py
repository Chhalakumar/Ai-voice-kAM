import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from .agent import build_graph

app = FastAPI(title="Snapdeal Virtual KAM")
graph = build_graph()


class ChatIn(BaseModel):
    session_id: str
    message: str


def _text(content) -> str:
    """Gemini may return a string or a list of content parts."""
    if isinstance(content, str):
        return content
    return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content).strip()


@app.post("/api/chat")
def chat(body: ChatIn):
    cfg = {"configurable": {"thread_id": body.session_id}}
    try:
        result = graph.invoke({"messages": [HumanMessage(content=body.message)]}, cfg)
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        print(traceback.format_exc())
        raise HTTPException(500, str(e))

    msgs = result["messages"]
    last_human = max(i for i, m in enumerate(msgs) if isinstance(m, HumanMessage))
    this_turn = msgs[last_human + 1:]

    results = {m.tool_call_id: m.content for m in this_turn if isinstance(m, ToolMessage)}
    trace = []
    for m in this_turn:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                raw = results.get(tc["id"])
                try:
                    out = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    out = raw
                trace.append({"tool": tc["name"], "args": tc["args"], "result": out})

    reply = _text(this_turn[-1].content) if this_turn else ""
    return {"reply": reply or "Sorry, could you say that again?", "trace": trace}


app.mount("/", StaticFiles(directory=Path(__file__).parent.parent / "public", html=True), name="static")
