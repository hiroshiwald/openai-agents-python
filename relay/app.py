"""A web page that runs a work loop between an OpenAI model and Claude.

One side directs the work. The other side does it, running real code in a
sandbox and reporting the real output. Each request runs exactly one turn, so
no single request approaches the host's request time limit. The browser owns
the loop and holds the transcript, so there is no database and no session state
on the server.

Set OPENAI_API_KEY and ANTHROPIC_API_KEY in the environment. Never in code.
"""

import logging
import os
from typing import Any, Literal

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from openai import OpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 240.0
MAX_TRANSCRIPT_CHARS = 250_000
MAX_CODE_OUTPUT_CHARS = 4_000
CODE_TOOL = {"type": "code_execution_20260521", "name": "code_execution"}
CODE_BETA = "code-execution-2025-08-25"

# Engineer models that accept adaptive thinking and an effort level. Older models and the
# Haiku line reject both, so offering them would make every turn fail. Add a prefix here
# when a newer model ships.
ENGINEER_PREFIXES = (
    "claude-fable-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
)

# OpenAI model families that do not take a text conversation.
DIRECTOR_EXCLUDES = (
    "audio",
    "embedding",
    "image",
    "moderation",
    "realtime",
    "transcribe",
    "tts",
)

DIRECTOR_PROTOCOL = """
You are the director in a two-person loop. An engineer works for you. The
engineer has a Python sandbox and can run real code.

Rules for this loop:
- Give the engineer one concrete task per message. Say what to build and what
  output you want back.
- Check every result the engineer reports. If a number or an output is missing,
  ask for it before you move on.
- Do not write the code yourself. Direct the work and review what comes back.
- When the goal is met, write DONE on the first line by itself, then summarise
  what was learned in a few sentences.
- Keep each message under 300 words.
""".strip()

ENGINEER_PROTOCOL = """
You are the engineer in a two-person loop. The director sends you one task at a
time. You do the work and report back.

Rules for this loop:
- Use the code execution tool to actually run the code. Report the real output.
- Never report a result you did not run. If a run fails, say what failed and
  show the error.
- Answer the current task. Do not repeat the history.
- Keep your prose under 400 words. Code and output do not count toward that.
""".strip()


class Turn(BaseModel):
    """One message from one side of the loop."""

    speaker: Literal["director", "engineer"]
    text: str


class TurnRequest(BaseModel):
    """Everything needed to produce the next single turn."""

    speaker: Literal["director", "engineer"]
    goal: str
    director_instructions: str = ""
    transcript: list[Turn] = Field(default_factory=list)
    openai_model: str = "gpt-6-astra"
    claude_model: str = "claude-sonnet-5"
    effort: Literal["low", "medium", "high"] = "medium"
    run_code: bool = True


class TurnResponse(BaseModel):
    """The message the model produced, plus anything it ran."""

    speaker: str
    text: str
    done: bool = False
    code_runs: list[str] = Field(default_factory=list)


app = FastAPI(title="Astra and Claude relay")


def _api_key(name: str) -> str:
    key = os.environ.get(name)
    if not key:
        raise HTTPException(status_code=503, detail=f"{name} is not set on the server.")
    return key


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [trimmed, {len(text) - limit} more characters]"


def _check_size(request: TurnRequest) -> None:
    total = sum(len(turn.text) for turn in request.transcript)
    if total > MAX_TRANSCRIPT_CHARS:
        raise HTTPException(
            status_code=413,
            detail="This run has grown too long. Copy the transcript and start a new run.",
        )


def _code_output(block: Any) -> str:
    """Read stdout, stderr, or an error code out of a code execution result."""
    content = getattr(block, "content", None)
    error = getattr(content, "error_code", None) or getattr(block, "error_code", None)
    stdout = (getattr(content, "stdout", "") or "").strip()
    stderr = (getattr(content, "stderr", "") or "").strip()
    pieces = []
    if error:
        pieces.append(f"error: {error}")
    if stdout:
        pieces.append(stdout)
    if stderr:
        pieces.append(f"stderr:\n{stderr}")
    return _clip("\n".join(pieces), MAX_CODE_OUTPUT_CHARS) if pieces else "(no output)"


def run_director(request: TurnRequest) -> TurnResponse:
    """Ask the OpenAI model for the next instruction to the engineer."""
    client = OpenAI(api_key=_api_key("OPENAI_API_KEY"), timeout=REQUEST_TIMEOUT)
    instructions = "\n\n".join(
        part for part in (request.director_instructions.strip(), DIRECTOR_PROTOCOL) if part
    )
    items: list[dict[str, str]] = [{"role": "user", "content": f"GOAL\n{request.goal}"}]
    for turn in request.transcript:
        if turn.speaker == "director":
            items.append({"role": "assistant", "content": turn.text})
        else:
            items.append({"role": "user", "content": f"ENGINEER REPORT\n{turn.text}"})
    response = client.responses.create(
        model=request.openai_model,
        instructions=instructions,
        input=items,
        max_output_tokens=4000,
    )
    text = (response.output_text or "").strip()
    return TurnResponse(
        speaker="director",
        text=text or "(the director returned nothing)",
        done=text.upper().startswith("DONE"),
    )


def run_engineer(request: TurnRequest) -> TurnResponse:
    """Ask Claude to do the current task, running code when it needs to."""
    client = anthropic.Anthropic(api_key=_api_key("ANTHROPIC_API_KEY"), timeout=REQUEST_TIMEOUT)
    messages: list[dict[str, str]] = [{"role": "user", "content": f"GOAL\n{request.goal}"}]
    for turn in request.transcript:
        if turn.speaker == "engineer":
            messages.append({"role": "assistant", "content": turn.text})
        else:
            messages.append({"role": "user", "content": f"DIRECTOR\n{turn.text}"})
    extra: dict[str, Any] = {}
    if request.run_code:
        extra = {"tools": [CODE_TOOL], "betas": [CODE_BETA]}
    response = client.beta.messages.create(
        model=request.claude_model,
        max_tokens=8000,
        system=ENGINEER_PROTOCOL,
        thinking={"type": "adaptive"},
        output_config={"effort": request.effort},
        messages=messages,
        **extra,
    )
    if response.stop_reason == "refusal":
        return TurnResponse(
            speaker="engineer",
            text="The engineer declined this task. Ask the director to reframe it.",
        )
    parts: list[str] = []
    code_runs: list[str] = []
    for block in response.content:
        if block.type == "text":
            parts.append(block.text)
        elif block.type == "bash_code_execution_tool_result":
            output = _code_output(block)
            code_runs.append(output)
            parts.append(f"[code output]\n{output}")
    text = "\n\n".join(part for part in parts if part.strip()).strip()
    return TurnResponse(
        speaker="engineer",
        text=text or "(the engineer returned nothing)",
        code_runs=code_runs,
    )


@app.get("/api/models")
def list_models() -> dict[str, list[str]]:
    """List the model names each key can reach, so nothing is hard coded."""
    result: dict[str, list[str]] = {"openai": [], "claude": []}
    try:
        client = OpenAI(api_key=_api_key("OPENAI_API_KEY"), timeout=30.0)
        result["openai"] = sorted(
            m.id
            for m in client.models.list()
            if m.id.startswith("gpt-") and not any(word in m.id for word in DIRECTOR_EXCLUDES)
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Could not list OpenAI models.")
    try:
        client2 = anthropic.Anthropic(api_key=_api_key("ANTHROPIC_API_KEY"), timeout=30.0)
        result["claude"] = [
            m.id for m in client2.models.list() if m.id.startswith(ENGINEER_PREFIXES)
        ]
    except HTTPException:
        raise
    except Exception:
        logger.exception("Could not list Claude models.")
    return result


@app.post("/api/turn")
def take_turn(request: TurnRequest) -> TurnResponse:
    """Run exactly one turn of the loop."""
    _check_size(request)
    try:
        if request.speaker == "director":
            return run_director(request)
        return run_engineer(request)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Turn failed for speaker %s.", request.speaker)
        raise HTTPException(status_code=502, detail="That turn failed. Try again.") from None


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Relay</title>
<style>
  :root {
    --bg: #ffffff; --fg: #1a1a1a; --muted: #5c5c5c; --line: #d8d8d8;
    --panel: #f6f6f6; --dir: #2b5fa8; --eng: #7a3ea8; --err: #b00020;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #16181c; --fg: #e8e8e8; --muted: #9aa0a8; --line: #333840;
      --panel: #1e2126; --dir: #7fa9e8; --eng: #c08ae8; --err: #ff8a95;
    }
  }
  * { box-sizing: border-box; }
  body {
    font-family: system-ui, sans-serif; background: var(--bg); color: var(--fg);
    margin: 0 auto; max-width: 52rem; padding: 1.5rem 1rem 4rem; line-height: 1.5;
  }
  h1 { font-size: 1.15rem; margin: 0 0 0.25rem; }
  p.sub { color: var(--muted); margin: 0 0 1.25rem; font-size: 0.9rem; }
  label { display: block; font-size: 0.8rem; color: var(--muted); margin: 0.75rem 0 0.25rem; }
  textarea, input, select {
    width: 100%; padding: 0.5rem; font: inherit; font-size: 0.9rem;
    background: var(--bg); color: var(--fg); border: 1px solid var(--line); border-radius: 6px;
  }
  textarea { resize: vertical; }
  details { border: 1px solid var(--line); border-radius: 6px; padding: 0.5rem 0.75rem; }
  summary { cursor: pointer; font-size: 0.85rem; color: var(--muted); }
  .row { display: grid; grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr)); gap: 0.5rem; }
  .buttons { display: flex; gap: 0.5rem; margin-top: 1rem; flex-wrap: wrap; }
  button {
    padding: 0.55rem 1.1rem; font: inherit; border-radius: 6px; cursor: pointer;
    border: 1px solid var(--line); background: var(--panel); color: var(--fg);
  }
  button.primary { background: var(--dir); border-color: var(--dir); color: #fff; }
  button[disabled] { opacity: 0.45; cursor: not-allowed; }
  #status { margin-top: 1rem; font-size: 0.85rem; color: var(--muted); min-height: 1.2rem; }
  #status.error { color: var(--err); }
  .turn { border-top: 1px solid var(--line); padding: 1rem 0; }
  .who { font-size: 0.75rem; letter-spacing: 0.06em; text-transform: uppercase; font-weight: 600; }
  .who.director { color: var(--dir); }
  .who.engineer { color: var(--eng); }
  .body { white-space: pre-wrap; overflow-wrap: anywhere; margin-top: 0.4rem; }
  .checkline { display: flex; align-items: center; gap: 0.4rem; margin-top: 1.25rem; }
  .checkline input { width: auto; }
  .checkline label { margin: 0; }
</style>
</head>
<body>
<h1>Relay</h1>
<p class="sub">The director plans and reviews. The engineer writes and runs the code.
They trade turns until the director says DONE or you press Stop.</p>

<label for="goal">What do you want done?</label>
<textarea id="goal" rows="3"
  placeholder="Example: measure how sorting time grows with list size in Python"></textarea>

<label for="instructions">Director instructions (your Astra prompt)</label>
<textarea id="instructions" rows="4"
  placeholder="Paste how you want the director to think and work. Leave blank for a plain director."
></textarea>

<details>
  <summary>Settings</summary>
  <div class="row">
    <div>
      <label for="openaiModel">Director model</label>
      <select id="openaiModel"></select>
    </div>
    <div>
      <label for="claudeModel">Engineer model</label>
      <select id="claudeModel"></select>
    </div>
    <div>
      <label for="effort">Engineer effort</label>
      <select id="effort">
        <option value="low">low, fastest</option>
        <option value="medium" selected>medium</option>
        <option value="high">high, slowest</option>
      </select>
    </div>
    <div>
      <label for="maxTurns">Turn limit</label>
      <input id="maxTurns" type="number" min="2" max="40" value="8">
    </div>
  </div>
  <div class="checkline">
    <input id="runCode" type="checkbox" checked>
    <label for="runCode">Let the engineer run code</label>
  </div>
</details>

<div class="buttons">
  <button id="start" class="primary" type="button">Start</button>
  <button id="continue" type="button" disabled>Continue</button>
  <button id="stop" type="button" disabled>Stop</button>
  <button id="copy" type="button" disabled>Copy transcript</button>
  <button id="clear" type="button" disabled>Clear</button>
</div>

<div id="status"></div>
<div id="log"></div>

<script>
const $ = (id) => document.getElementById(id);
let transcript = [];
let running = false;

function store(key, value) {
  try { localStorage.setItem(key, value); } catch (e) { /* private mode */ }
}
function recall(key, fallback) {
  try { return localStorage.getItem(key) ?? fallback; } catch (e) { return fallback; }
}

for (const id of ["goal", "instructions", "openaiModel", "claudeModel", "effort", "maxTurns"]) {
  const saved = recall("relay." + id, null);
  if (saved !== null) $(id).value = saved;
  $(id).addEventListener("change", () => store("relay." + id, $(id).value));
  $(id).addEventListener("input", () => store("relay." + id, $(id).value));
}

function setStatus(text, isError) {
  $("status").textContent = text;
  $("status").className = isError ? "error" : "";
}

function addTurn(turn) {
  const wrap = document.createElement("div");
  wrap.className = "turn";
  const who = document.createElement("div");
  who.className = "who " + turn.speaker;
  who.textContent = turn.speaker === "director" ? "Director" : "Engineer";
  const body = document.createElement("div");
  body.className = "body";
  body.textContent = turn.text;
  wrap.append(who, body);
  $("log").append(wrap);
  wrap.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function setButtons() {
  $("start").disabled = running;
  $("stop").disabled = !running;
  $("continue").disabled = running || transcript.length === 0;
  $("copy").disabled = transcript.length === 0;
  $("clear").disabled = running || transcript.length === 0;
}

async function loop() {
  running = true;
  setButtons();
  const limit = Math.max(2, Math.min(40, parseInt($("maxTurns").value, 10) || 8));
  let speaker = transcript.length === 0
    ? "director"
    : (transcript[transcript.length - 1].speaker === "director" ? "engineer" : "director");

  for (let i = 0; i < limit && running; i++) {
    setStatus((speaker === "director" ? "Director" : "Engineer") + " is working. Turn "
      + (transcript.length + 1) + ".");
    let data;
    try {
      const response = await fetch("/api/turn", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          speaker,
          goal: $("goal").value.trim(),
          director_instructions: $("instructions").value,
          transcript,
          openai_model: $("openaiModel").value,
          claude_model: $("claudeModel").value,
          effort: $("effort").value,
          run_code: $("runCode").checked,
        }),
      });
      data = await response.json();
      if (!response.ok) {
        setStatus(data.detail || "That turn failed.", true);
        break;
      }
    } catch (error) {
      setStatus("Could not reach the server. Check your connection and press Continue.", true);
      break;
    }
    transcript.push({ speaker: data.speaker, text: data.text });
    addTurn(data);
    setButtons();
    if (data.done) { setStatus("The director marked this done."); running = false; break; }
    speaker = speaker === "director" ? "engineer" : "director";
  }

  if (running) setStatus("Turn limit reached. Press Continue to keep going.");
  running = false;
  setButtons();
}

$("start").addEventListener("click", () => {
  if (!$("goal").value.trim()) { setStatus("Type what you want done first.", true); return; }
  transcript = [];
  $("log").textContent = "";
  loop();
});
$("continue").addEventListener("click", () => loop());
$("stop").addEventListener("click", () => {
  running = false;
  setStatus("Stopping after the current turn finishes.");
});
$("clear").addEventListener("click", () => {
  transcript = [];
  $("log").textContent = "";
  setStatus("");
  setButtons();
});
$("copy").addEventListener("click", async () => {
  const text = transcript
    .map((t) => (t.speaker === "director" ? "DIRECTOR" : "ENGINEER") + "\\n" + t.text)
    .join("\\n\\n----\\n\\n");
  try {
    await navigator.clipboard.writeText(text);
    setStatus("Transcript copied.");
  } catch (error) {
    setStatus("Could not copy. Select the text and copy it by hand.", true);
  }
});

function fillModels(select, names, preferred) {
  const saved = recall("relay." + select.id, null);
  const list = names.length ? names : [preferred];
  select.textContent = "";
  for (const name of list) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    select.append(option);
  }
  const wanted = (saved && list.includes(saved)) ? saved
    : (list.includes(preferred) ? preferred : list[0]);
  select.value = wanted;
}

fetch("/api/models")
  .then((response) => response.ok ? response.json() : { openai: [], claude: [] })
  .then((data) => {
    fillModels($("openaiModel"), data.openai || [], "gpt-6-astra");
    fillModels($("claudeModel"), data.claude || [], "claude-sonnet-5");
  })
  .catch(() => {
    fillModels($("openaiModel"), [], "gpt-6-astra");
    fillModels($("claudeModel"), [], "claude-sonnet-5");
  });

setButtons();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the single-page relay console."""
    return PAGE
