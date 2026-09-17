"""Advance relay runs that are stored as GitHub issues.

One issue is one run. The issue body holds the settings and a hidden state
line. Each comment is one turn. A scheduled workflow calls this script, which
moves every open run forward by a few turns and then exits. Nothing stays
running between calls, so nothing needs a browser or a server.

Every model call is checked against the run's dollar budget before it is sent,
using an upper-bound estimate. Real spend is added up from the token counts the
providers report and written back to the issue.
"""

import json
import os
import re
import sys
import time
from typing import Any

import anthropic
import requests
from openai import OpenAI

from pricing import cost_of, worst_case_cost

API = "https://api.github.com"
TURNS_PER_INVOCATION = 4
DIRECTOR_MAX_OUTPUT = 4000
ENGINEER_MAX_OUTPUT = 8000
REQUEST_TIMEOUT = 300.0
MAX_CODE_OUTPUT_CHARS = 4000
DEFAULT_BUDGET = 2.00
BUDGET_CEILING = 50.00
RUN_LABEL = "relay"
STATE_RE = re.compile(r"<!--\s*relay-state\s*(\{.*?\})\s*-->", re.DOTALL)
MARKER_RE = re.compile(r"<!--\s*relay:(director|engineer|notice)\s*-->")
FOOTER_RE = re.compile(r"\n+_Turn \d+[^\n]*_\s*$")

ENGINEER_PREFIXES = (
    "claude-fable-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
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
- If you need a decision from the human who set the goal, write ASK on the first
  line by itself, then your question. The loop will stop and wait for a reply.
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


def log(message: str) -> None:
    print(message, flush=True)


class Stop(Exception):
    """Raised to end work on one run without ending the whole invocation."""


# --------------------------------------------------------------------------
# GitHub
# --------------------------------------------------------------------------


class Repo:
    """The few GitHub REST calls this script needs."""

    def __init__(self, token: str, full_name: str) -> None:
        self.base = f"{API}/repos/{full_name}"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )

    def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        for attempt in range(3):
            response = self.session.request(method, f"{self.base}{path}", timeout=30, **kwargs)
            if response.status_code < 500:
                response.raise_for_status()
                return response.json() if response.content else None
            time.sleep(2**attempt)
        response.raise_for_status()

    def open_runs(self) -> list[dict[str, Any]]:
        issues = self._call("GET", f"/issues?state=open&labels={RUN_LABEL}&per_page=20")
        return [issue for issue in issues if "pull_request" not in issue]

    def comments(self, number: int) -> list[dict[str, Any]]:
        return self._call("GET", f"/issues/{number}/comments?per_page=100")

    def comment(self, number: int, body: str) -> None:
        self._call("POST", f"/issues/{number}/comments", json={"body": body})

    def set_body(self, number: int, body: str) -> None:
        self._call("PATCH", f"/issues/{number}", json={"body": body})

    def set_labels(self, number: int, labels: list[str]) -> None:
        try:
            self._call("PUT", f"/issues/{number}/labels", json={"labels": labels})
        except requests.HTTPError:
            log(f"  could not set labels on #{number}")

    def close(self, number: int) -> None:
        self._call("PATCH", f"/issues/{number}", json={"state": "closed"})


# --------------------------------------------------------------------------
# Issue body: settings, state
# --------------------------------------------------------------------------


def parse_form(body: str) -> dict[str, str]:
    """Read the fields of a GitHub issue form out of the rendered issue body."""
    fields: dict[str, str] = {}
    heading = None
    lines: list[str] = []
    for line in body.splitlines():
        if line.startswith("### "):
            if heading:
                fields[heading] = "\n".join(lines).strip()
            heading = line[4:].strip().lower()
            lines = []
        elif heading:
            lines.append(line)
    if heading:
        fields[heading] = "\n".join(lines).strip()
    return {k: ("" if v == "_No response_" else v) for k, v in fields.items()}


def read_state(body: str) -> dict[str, Any]:
    match = STATE_RE.search(body)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            log("  state line was unreadable, starting fresh")
    return {"turn": 0, "spent": 0.0, "status": "running", "checkin_turn": 0}


def write_state(body: str, state: dict[str, Any]) -> str:
    line = f"<!-- relay-state {json.dumps(state, separators=(',', ':'))} -->"
    if STATE_RE.search(body):
        return STATE_RE.sub(lambda _: line, body, count=1)
    return f"{body.rstrip()}\n\n{line}\n"


def number_field(fields: dict[str, str], name: str, default: float, cap: float) -> float:
    raw = (fields.get(name) or "").strip().lstrip("$")
    try:
        value = float(raw)
    except ValueError:
        return default
    if value <= 0:
        return default
    return min(value, cap)


# --------------------------------------------------------------------------
# Transcript
# --------------------------------------------------------------------------


def build_transcript(comments: list[dict[str, Any]]) -> tuple[list[dict[str, str]], bool]:
    """Turn the issue's comments into a transcript.

    Returns the transcript and whether a human spoke after the last bot comment.
    """
    transcript: list[dict[str, str]] = []
    human_spoke_last = False
    for item in comments:
        body = item.get("body") or ""
        match = MARKER_RE.search(body)
        text = MARKER_RE.sub("", body).strip()
        text = FOOTER_RE.sub("", text).strip()
        if match:
            if match.group(1) != "notice":
                transcript.append({"speaker": match.group(1), "text": text})
            human_spoke_last = False
        elif text:
            transcript.append({"speaker": "human", "text": text})
            human_spoke_last = True
    return transcript, human_spoke_last


def transcript_chars(transcript: list[dict[str, str]]) -> int:
    return sum(len(turn["text"]) for turn in transcript)


# --------------------------------------------------------------------------
# Model calls
# --------------------------------------------------------------------------


def clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [trimmed, {len(text) - limit} more characters]"


def code_output(block: Any) -> str:
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
    return clip("\n".join(pieces), MAX_CODE_OUTPUT_CHARS) if pieces else "(no output)"


def call_director(
    model: str, goal: str, instructions: str, transcript: list[dict[str, str]]
) -> tuple[str, float]:
    client = OpenAI(timeout=REQUEST_TIMEOUT)
    system = "\n\n".join(part for part in (instructions.strip(), DIRECTOR_PROTOCOL) if part)
    items: list[dict[str, str]] = [{"role": "user", "content": f"GOAL\n{goal}"}]
    for turn in transcript:
        if turn["speaker"] == "director":
            items.append({"role": "assistant", "content": turn["text"]})
        elif turn["speaker"] == "engineer":
            items.append({"role": "user", "content": f"ENGINEER REPORT\n{turn['text']}"})
        else:
            items.append({"role": "user", "content": f"MESSAGE FROM THE HUMAN\n{turn['text']}"})
    response = client.responses.create(
        model=model,
        instructions=system,
        input=items,
        max_output_tokens=DIRECTOR_MAX_OUTPUT,
    )
    usage = getattr(response, "usage", None)
    spent = cost_of(
        model,
        getattr(usage, "input_tokens", 0) or 0,
        getattr(usage, "output_tokens", 0) or 0,
    )
    return (response.output_text or "").strip(), spent


def call_engineer(
    model: str, goal: str, transcript: list[dict[str, str]], effort: str
) -> tuple[str, float]:
    client = anthropic.Anthropic(timeout=REQUEST_TIMEOUT)
    messages: list[dict[str, str]] = [{"role": "user", "content": f"GOAL\n{goal}"}]
    for turn in transcript:
        if turn["speaker"] == "engineer":
            messages.append({"role": "assistant", "content": turn["text"]})
        elif turn["speaker"] == "director":
            messages.append({"role": "user", "content": f"DIRECTOR\n{turn['text']}"})
        else:
            messages.append({"role": "user", "content": f"MESSAGE FROM THE HUMAN\n{turn['text']}"})
    response = client.beta.messages.create(
        model=model,
        max_tokens=ENGINEER_MAX_OUTPUT,
        system=ENGINEER_PROTOCOL,
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
        messages=messages,
        tools=[{"type": "code_execution_20260521", "name": "code_execution"}],
        betas=["code-execution-2025-08-25"],
    )
    usage = response.usage
    spent = cost_of(model, usage.input_tokens or 0, usage.output_tokens or 0)
    if response.stop_reason == "refusal":
        return "The engineer declined this task. Ask the director to reframe it.", spent
    parts: list[str] = []
    for block in response.content:
        if block.type == "text":
            parts.append(block.text)
        elif block.type == "bash_code_execution_tool_result":
            parts.append(f"```\n{code_output(block)}\n```")
    return "\n\n".join(part for part in parts if part.strip()).strip(), spent


# --------------------------------------------------------------------------
# Advancing one run
# --------------------------------------------------------------------------


def next_speaker(transcript: list[dict[str, str]]) -> str:
    for turn in reversed(transcript):
        if turn["speaker"] == "director":
            return "engineer"
        if turn["speaker"] == "engineer":
            return "director"
    return "director"


def trailing_human_text(transcript: list[dict[str, str]]) -> str:
    for turn in reversed(transcript):
        if turn["speaker"] == "human":
            return turn["text"].strip()
        return ""
    return ""


def apply_command(text: str, state: dict[str, Any]) -> str | None:
    """Read a human command out of a comment. Returns a status, or None."""
    head = text.upper()
    if head.startswith("STOP"):
        return "stopped"
    if head.startswith("PAUSE"):
        return "waiting"
    if head.startswith("BUDGET"):
        match = re.search(r"[\d.]+", text)
        if match:
            state["budget_override"] = min(float(match.group()), BUDGET_CEILING)
    return None


def advance(repo: Repo, issue: dict[str, Any]) -> None:
    number = issue["number"]
    body = issue.get("body") or ""
    fields = parse_form(body)
    state = read_state(body)
    log(f"issue #{number}: status={state['status']} turn={state['turn']}")

    goal = fields.get("goal", "").strip()
    instructions = fields.get("director instructions", "")
    checkin_every = int(number_field(fields, "check in with me every n turns", 4, 100))
    turn_limit = int(number_field(fields, "turn limit", 20, 200))
    director_model = (fields.get("director model") or "gpt-6-astra").strip()
    engineer_model = (fields.get("engineer model") or "claude-sonnet-5").strip()
    effort = (fields.get("engineer effort") or "medium").strip().lower()
    if effort not in ("low", "medium", "high"):
        effort = "medium"
    if not engineer_model.startswith(ENGINEER_PREFIXES):
        engineer_model = "claude-sonnet-5"

    transcript, human_spoke_last = build_transcript(repo.comments(number))

    # Read the newest human comment before anything else, so a command can
    # revive a run that has already stopped.
    command = trailing_human_text(transcript)
    command_status = apply_command(command, state)
    budget = float(
        state.get("budget_override")
        or number_field(fields, "budget in us dollars", DEFAULT_BUDGET, BUDGET_CEILING)
    )

    def save(status: str) -> None:
        state["status"] = status
        repo.set_body(number, write_state(body, state))
        repo.set_labels(number, [RUN_LABEL, f"relay:{status}"])

    def money() -> str:
        return f"${state['spent']:.2f} of ${budget:.2f}"

    def finish(status: str, reason: str, message: str) -> None:
        state["stop_reason"] = reason
        repo.comment(number, f"<!-- relay:notice -->\n{message}\n\nSpent {money()}.")
        save(status)

    if not goal:
        finish("stopped", "no-goal", "This run has no goal, so it cannot start.")
        return

    if command_status == "stopped":
        finish("stopped", "human", "Stopped, as you asked. This run will not start again.")
        return

    if state["status"] == "stopped":
        return

    if state["status"] == "done":
        if not human_spoke_last:
            return
        revivable = state.get("stop_reason") in ("budget", "turns")
        asked = command.upper().startswith(("CONTINUE", "BUDGET"))
        if revivable and asked and state["spent"] < budget and state["turn"] < turn_limit:
            log("  reviving a stopped run")
            state["checkin_turn"] = state["turn"]
            state["status"] = "running"
        else:
            repo.comment(
                number,
                "<!-- relay:notice -->\nThis run has finished, so it did not act on that "
                "comment.\n\nTo carry on, raise the limit that stopped it and comment "
                "`CONTINUE`. To raise the budget and carry on in one step, comment "
                "`BUDGET 5`. To start something new, open a new run.",
            )
            return

    if command_status == "waiting":
        finish("waiting", "human", "Paused. Comment anything to start again.")
        return

    if state["status"] == "waiting":
        if not human_spoke_last:
            log("  still waiting for a human reply")
            return
        state["checkin_turn"] = state["turn"]
        state["status"] = "running"

    save("running")

    for _ in range(TURNS_PER_INVOCATION):
        if state["turn"] >= turn_limit:
            finish(
                "done",
                "turns",
                f"Reached the turn limit of {turn_limit}. Raise the turn limit in the "
                "description above, then comment `CONTINUE`.",
            )
            return

        speaker = next_speaker(transcript)
        model = director_model if speaker == "director" else engineer_model
        cap = DIRECTOR_MAX_OUTPUT if speaker == "director" else ENGINEER_MAX_OUTPUT
        prompt_chars = transcript_chars(transcript) + len(goal) + len(instructions) + 1500
        estimate = worst_case_cost(model, prompt_chars, cap)
        if state["spent"] + estimate > budget:
            finish(
                "done",
                "budget",
                f"Stopping to stay inside the budget. The next turn could cost up to "
                f"${estimate:.2f}, which would go over. Comment `BUDGET 5` to raise the "
                "limit to five dollars and carry on.",
            )
            return

        log(f"  turn {state['turn'] + 1}: {speaker} on {model}")
        if speaker == "director":
            text, spent = call_director(model, goal, instructions, transcript)
        else:
            text, spent = call_engineer(model, goal, transcript, effort)
        text = text or "(this turn produced nothing)"

        state["spent"] = round(state["spent"] + spent, 6)
        state["turn"] += 1
        transcript.append({"speaker": speaker, "text": text})

        title = "Director" if speaker == "director" else "Engineer"
        footer = f"_Turn {state['turn']} of {turn_limit} \u00b7 spent {money()}_"
        repo.comment(number, f"<!-- relay:{speaker} -->\n**{title}**\n\n{text}\n\n{footer}")
        save("running")

        head = text.upper()
        if speaker == "director" and head.startswith("DONE"):
            finish("done", "director", "The director marked this run finished.")
            return
        if speaker == "director" and head.startswith("ASK"):
            state["checkin_turn"] = state["turn"]
            finish("waiting", "ask", "The director has a question for you. Reply to carry on.")
            return
        if state["turn"] - state["checkin_turn"] >= checkin_every:
            state["checkin_turn"] = state["turn"]
            finish(
                "waiting",
                "checkin",
                f"Checking in after {checkin_every} turns. Reply with guidance to carry on, "
                "`STOP` to end the run, or `BUDGET 5` to raise the limit.",
            )
            return

    save("running")
    log(f"  paused after {TURNS_PER_INVOCATION} turns, will resume on the next run")


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN")
    full_name = os.environ.get("GITHUB_REPOSITORY")
    if not token or not full_name:
        log("GITHUB_TOKEN and GITHUB_REPOSITORY must be set.")
        return 1
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        if not os.environ.get(key):
            log(f"{key} is not set. Add it under Settings, Secrets and variables, Actions.")
            return 1

    repo = Repo(token, full_name)
    runs = repo.open_runs()
    log(f"found {len(runs)} open run(s)")
    for issue in runs:
        try:
            advance(repo, issue)
        except Exception as error:
            log(f"issue #{issue['number']} failed: {type(error).__name__}: {error}")
            try:
                repo.comment(
                    issue["number"],
                    "<!-- relay:notice -->\nThis turn failed with a "
                    f"`{type(error).__name__}`. The run will try again on the next pass. "
                    "Open the Actions tab for the full error.",
                )
            except Exception:
                log("  could not post the failure notice")
    return 0


if __name__ == "__main__":
    sys.exit(main())
