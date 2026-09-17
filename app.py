"""Minimal FastAPI web app that runs an OpenAI Agents SDK agent.

Run locally with `uvicorn app:app --reload`, or deploy to Vercel, which detects
the `app` ASGI instance in this file without extra configuration. Set the
`OPENAI_API_KEY` environment variable before sending a message.
"""

import logging
import os
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agents import Agent, Runner
from agents.decorators import tool

logger = logging.getLogger(__name__)


@tool
def current_time() -> str:
    """Return the current UTC date and time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


agent = Agent(
    name="Assistant",
    instructions=(
        "You are a helpful assistant. Answer in one or two short sentences. "
        "Use the current_time tool when the user asks about the date or the time."
    ),
    tools=[current_time],
)

app = FastAPI(title="Agents SDK demo")


class ChatRequest(BaseModel):
    """A single user message."""

    message: str


class ChatResponse(BaseModel):
    """The agent reply for one user message."""

    reply: str


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agents SDK demo</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0 auto; max-width: 40rem; padding: 2rem 1rem; }
  h1 { font-size: 1.25rem; }
  form { display: flex; gap: 0.5rem; }
  input { flex: 1; padding: 0.6rem; font-size: 1rem; }
  button { padding: 0.6rem 1rem; font-size: 1rem; cursor: pointer; }
  button[disabled] { cursor: progress; opacity: 0.6; }
  #reply { margin-top: 1.5rem; white-space: pre-wrap; min-height: 3rem; }
  .error { color: #b00020; }
</style>
</head>
<body>
<h1>Agents SDK demo</h1>
<p>Ask a question. Try &ldquo;What time is it?&rdquo; to make the agent call the tool.</p>
<form id="chat">
  <input id="message" name="message" autocomplete="off" placeholder="Type a message" required>
  <button id="send" type="submit">Send</button>
</form>
<div id="reply"></div>
<script>
  const form = document.getElementById("chat");
  const input = document.getElementById("message");
  const send = document.getElementById("send");
  const reply = document.getElementById("reply");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = input.value.trim();
    if (!message) return;
    send.disabled = true;
    reply.className = "";
    reply.textContent = "Thinking...";
    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
      const data = await response.json();
      if (response.ok) {
        reply.textContent = data.reply;
      } else {
        reply.className = "error";
        reply.textContent = data.detail || "The request failed.";
      }
    } catch (error) {
      reply.className = "error";
      reply.textContent = "The request failed.";
    } finally {
      send.disabled = false;
    }
  });
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Serve the single-page chat form."""
    return PAGE


@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """Run the agent on one user message and return its final output."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not set on the server.")
    try:
        result = await Runner.run(agent, request.message)
    except Exception:
        logger.exception("Agent run failed.")
        raise HTTPException(status_code=502, detail="The agent run failed.") from None
    return ChatResponse(reply=str(result.final_output))
