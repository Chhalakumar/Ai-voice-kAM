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
        error_msg = str(e)
        print(f"ERROR: {error_msg}")
        print(traceback.format_exc())
        
        # Check if it's a quota error
        if "quota" in error_msg.lower() or "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return {
                "reply": "I apologize, but I've reached my daily quota limit. Please try again tomorrow or use a different API key.",
                "trace": [],
                "error": "quota_exceeded"
            }
        
        # Return JSON error instead of raising HTTPException
        return {
            "reply": f"Sorry, I encountered an error: {error_msg[:100]}. Please try again.",
            "trace": [],
            "error": "server_error"
        }

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


class UpdateDataIn(BaseModel):
    last_sales: int
    current_sales: int
    last_orders: int
    current_orders: int
    last_cancel_rate: float
    current_cancel_rate: float
    last_asp: int
    current_asp: int
    last_oos: int
    current_oos: int


@app.post("/api/update-data")
def update_data(body: UpdateDataIn):
    """Update mock seller data for testing different scenarios."""
    from .data import LAST, CURRENT
    
    # Update LAST month data
    LAST["sales"] = body.last_sales
    LAST["orders"] = body.last_orders
    LAST["cancel_rate"] = body.last_cancel_rate / 100  # Convert percentage to decimal
    LAST["asp"] = body.last_asp
    LAST["oos"] = body.last_oos
    
    # Update CURRENT month data
    CURRENT["sales"] = body.current_sales
    CURRENT["orders"] = body.current_orders
    CURRENT["cancel_rate"] = body.current_cancel_rate / 100
    CURRENT["asp"] = body.current_asp
    CURRENT["oos"] = body.current_oos
    
    return {
        "success": True,
        "message": "Mock data updated successfully",
        "data": {
            "last": LAST,
            "current": CURRENT
        }
    }


@app.get("/api/get-data")
def get_data():
    """Get current mock seller data."""
    from .data import LAST, CURRENT
    return {
        "last": LAST,
        "current": CURRENT
    }


app.mount("/", StaticFiles(directory=Path(__file__).parent.parent / "public", html=True), name="static")
