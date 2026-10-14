"""AI assistant for the challenges that allow one: a small, fast chat model that has read the statement.

The models are served for free by OpenRouter. Each user brings their own key (https://openrouter.ai/keys):
- on Kaggle: Add-ons -> Secrets -> add a secret named OPENROUTER_API_KEY, then tick it for the notebook;
- locally: an OPENROUTER_API_KEY environment variable, or a line OPENROUTER_API_KEY=... in a .env file
  in the notebook's folder or one of its parents.

In the notebook:
    ask("question")   the latest run of your solution cell and its sample results are attached automatically
    fix()             as the LAST line of your solution cell: the cell is replaced by a corrected version
    undo()            as the last line of your solution cell: restores the version before the last fix()
    reset()           forgets the conversation

What the model sees is limited to: the statement and the sample tests (the challenge's context file), the text
of the solution cell (the one starting with the RESOLUTION header), and the results of running my_solution on
the sample tests. No other cell is ever read, in particular not the proposed solution.

Standard library only. Every notebook loads this file from the repo, so editing MODELS here updates them all.
"""

import ast
import contextlib
import copy
import ctypes
import hashlib
import io
import json
import os
import re
import sys
import threading
import traceback
import urllib.error
import urllib.request
from pathlib import Path

KEY_NAME = "OPENROUTER_API_KEY"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Tried in order: a model that is rate-limited, removed or failing falls through to the next one.
# Free, mid-sized and fast, with reasoning switched off: genuinely helpful, but not meant to one-shot the
# problem (`python -m cfnb oneshot <pid>` measures that). Best first; the Gemmas are often rate-limited upstream.
MODELS = [
    "dots-studio/dots-3-note-preview:free",  # strongest under the 8k window (~1280 one-shot), never cut off
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3.5-lightning:free",
    "poolside/laguna-xs-2.1:free",
    "cohere/north-mini-code:free",  # slow and thinks aloud: last resort
]
# The assistant's context window, whatever the model: the whole conversation (system prompt with the statement,
# questions, answers) plus the next answer must fit in it. Each answer gets the room that is left (at most
# MAX_TOKENS). After an answer, a conversation over PRUNE_AT of the window loses its oldest question/answer pairs,
# PRUNE_SHARE of the window's worth; the same happens before sending when less than MIN_ANSWER_ROOM is left.
CONTEXT_WINDOW = 8000
MAX_TOKENS = 8000
MIN_ANSWER_ROOM = 1000
PRUNE_AT, PRUNE_SHARE = 0.9, 0.1
CONTEXT_FULL = re.compile(r"context (length|window)|maximum context|too many tokens|prompt is too long", re.I)

RESOLUTION_HEADER = "######################### RESOLUTION"  # first line of the notebook's solution cell
COMMAND_LINE = re.compile(r"^\s*(?:assistant\.)?(?:fix|undo)\(.*\)\s*(?:#.*)?$")

SYSTEM_PROMPT = """You are the assistant of a programmer solving a Codeforces problem in a Python notebook.
They implement `my_solution`, which solves ONE test case: its arguments are already parsed and it returns \
the answer (see the interface below). It does not read input or print anything.
Their messages may include their solution cell and the results of running it on the sample tests: that is all \
you see of their notebook.
Be concise and accurate. Answer in English (unless asked in another language), in Markdown, with code in \
```python blocks.
Answer exactly what is asked: explain, hint or debug, and write a full solution only if explicitly asked for one.
When code is shown, point to concrete bugs, with a small failing input when you can. If you are unsure, say so.

{context}"""

FIX_REQUEST = """Fix the bugs in my solution cell below.
Keep my approach, names and structure, and change as little as possible. If the approach itself cannot work, \
say so in one or two sentences and fix what you can anyway.
Don't think aloud: list the bugs in at most five short bullet points. Then give the complete corrected cell in a single ```python block, starting with \
its first comment line, without any call to fix() or undo()."""

NO_KEY = (f"🔑 No OpenRouter key found. Get a free one at https://openrouter.ai/keys, then either add a Kaggle "
          f"secret named {KEY_NAME} (Add-ons → Secrets, and tick it for this notebook) or, locally, put "
          f"{KEY_NAME}=... in a .env file. Rerun this cell afterwards (or paste the key when ask() prompts for it).")

NOT_IN_CELL = ("{name}() only works as the last line of your solution cell (the one starting with "
               "\"######################### RESOLUTION\"): add it there and run that cell.")

_current = None


def find_key():
    """Return (key, where it was found), or (None, None)."""
    key = os.environ.get(KEY_NAME, "").strip()
    if key:
        return key, "environment variable"
    try:
        from kaggle_secrets import UserSecretsClient
        key = (UserSecretsClient().get_secret(KEY_NAME) or "").strip()
        if key:
            return key, "Kaggle secret"
    except Exception:  # not on Kaggle, or the secret is not attached to this notebook
        pass
    for folder in (Path.cwd(), *Path.cwd().parents):
        env = folder / ".env"
        if env.is_file():
            for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
                name, sep, value = line.partition("=")
                if sep and name.strip().removeprefix("export ").strip() == KEY_NAME and value.strip():
                    return value.strip().strip("'\""), ".env file"
    return None, None


def _chars(messages):
    return sum(len(m["content"]) for m in messages)


# ------------------------------- what the model may see of the notebook -------------------------------

def _ipython():
    try:
        from IPython import get_ipython
        return get_ipython()
    except ImportError:
        return None


def _is_solution_cell(text):
    return text.lstrip().startswith(RESOLUTION_HEADER)


def _strip_commands(text):
    return "\n".join(line for line in text.splitlines() if not COMMAND_LINE.match(line)).rstrip() + "\n"


def solution_cell():
    """Text of the latest run of the solution cell (without fix()/undo() lines), or None if it never ran."""
    ip = _ipython()
    history = ip.user_ns.get("In", []) if ip else []
    for text in reversed(history):
        if _is_solution_cell(text):
            return _strip_commands(text)
    return None


def _call(fn, args, seconds):
    """Run fn(*args) in a thread with a time limit. Returns ("ok", value) / ("error", exc) / ("timeout", None)."""
    box = {}

    def target():
        try:
            with contextlib.redirect_stdout(io.StringIO()):  # the user's debug prints stay out of the answer
                box["value"] = fn(*args)
        except BaseException as e:  # noqa: B036 - also catches the stop request below
            box["error"] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():  # ask the thread to stop (works for pure-Python loops)
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread.ident), ctypes.py_object(SystemExit))
        thread.join(1)
        return "timeout", None
    return ("error", box["error"]) if "error" in box else ("ok", box.get("value"))


def _where(error):
    """'line 7: x = a[i]' for the deepest frame of the user's code (line numbers of their cell)."""
    frames = [f for f in traceback.extract_tb(error.__traceback__) if Path(f.filename).name != "assistant.py"]
    if not frames:
        return ""
    frame = frames[-1]
    return f" (line {frame.lineno}: `{(frame.line or '').strip()}`)"


def _short(value, limit=150):
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "…"


def sample_report():
    """Run my_solution on the notebook's sample tests and describe the results, or return None."""
    ns = _ipython().user_ns if _ipython() else {}
    solve, samples, check = ns.get("my_solution"), ns.get("samples"), ns.get("check_solution")
    if not callable(solve) or not samples or not callable(check):
        return None
    utils = sys.modules.get(getattr(check, "__module__", ""))
    unique = getattr(utils, "ANSWER_KIND", "unique") == "unique"
    seconds = min(10.0, max(2.0, 2 * float(ns.get("TIME_LIMIT", 2))))
    lines = ["Results of my `my_solution` on the sample tests (run just now):"]
    for i, case in enumerate(samples, 1):
        call = f"my_solution({', '.join(_short(v, 80) for v in case)})"
        status, value = _call(solve, copy.deepcopy(list(case)), seconds)
        if status == "timeout":
            lines.append(f"- Sample {i} `{call}`: still running after {seconds:g}s (infinite loop?)")
            break  # the next samples would most likely hang too
        if status == "error":
            lines.append(f"- Sample {i} `{call}`: raised {type(value).__name__}: {value}{_where(value)}")
            continue
        try:
            ok, _, _, message = check(case, value, None, 0, 0)
        except Exception as e:  # noqa: BLE001
            ok, message = False, str(e)
        if ok:
            lines.append(f"- Sample {i} `{call}`: correct")
            continue
        detail = f"returned `{_short(value)}`"
        entry = utils._get_expected(case) if unique and hasattr(utils, "_get_expected") else None
        detail += f", expected `{_short(entry[1])}`" if entry else ""
        lines.append(f"- Sample {i} `{call}`: {detail}: wrong" + (f" ({message})" if message else ""))
    return "\n".join(lines)


def _code_block(answer):
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", answer, re.S)
    for block in reversed(blocks):
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue
        if any(isinstance(n, ast.FunctionDef) and n.name == "my_solution" for n in tree.body):
            return block
    return None


# --------------------------------------------- chat ---------------------------------------------

class _Output:
    """The answer streams as plain text, then (in a notebook) the cell's output is replaced by the rendered
    Markdown. Only print, clear_output and display: updating a display in place is ignored by VS Code."""

    def __init__(self):
        self.text = ""
        self.notebook = _ipython() is not None
        print("⏳ ", end="", flush=True)

    def add(self, piece):
        self.text += piece
        print(piece, end="", flush=True)

    def finish(self, note):
        full = self.text + ("\n\n" if self.text else "") + note
        if not self.notebook:
            print(("\n\n" if self.text else "") + note)
            return
        from IPython.display import Markdown, clear_output, display
        clear_output(wait=True)
        display(Markdown(full))


class _ModelFailed(Exception):
    pass


class Assistant:
    def __init__(self, context):
        self.context = context
        self.key, self.key_source = find_key()
        self.backup = None  # the solution cell before the last fix()
        self.truncated = False  # did the last answer stop at MAX_TOKENS?
        self.usage = None  # token usage reported for the last answer
        self.reset()

    def reset(self):
        """Forget the conversation."""
        self.history = [{"role": "system", "content": SYSTEM_PROMPT.format(context=self.context)}]
        self.sent = None  # hash of the solution cell last shown to the model
        self.used = 0  # tokens of the conversation after the last answer

    def ask(self, question, *_ignored):
        """Ask a question. The solution cell and its sample results are attached when they changed."""
        cell = solution_cell()
        parts = [question]
        if cell is None:
            parts.append("(I haven't run my solution cell yet.)")
        elif self._digest(cell) == self.sent:
            parts.append("(My solution cell is unchanged since I last showed it.)")
        else:
            parts += [f"My solution cell:\n```python\n{cell}```", sample_report() or ""]
        self._chat("\n\n".join(p for p in parts if p), cell)

    def fix(self, hint=""):
        """Replace the solution cell (the one this is called from) by a corrected version."""
        ip = _ipython()
        current = ip.user_ns.get("In", [""])[-1] if ip else ""
        if not _is_solution_cell(current):
            print(NOT_IN_CELL.format(name="fix"))
            return
        cell = _strip_commands(current)
        parts = [FIX_REQUEST, f"Hint from me: {hint}" if hint else "",
                 f"My solution cell:\n```python\n{cell}```", sample_report() or ""]
        answer = self._chat("\n\n".join(p for p in parts if p), cell)
        if answer is None:
            return
        code = _code_block(answer)
        if code is None:
            print("No complete my_solution in the answer: your cell was not changed. "
                  "Try fix() again (another model may answer), or ask() about the bug.")
            return
        if not _is_solution_cell(code):
            code = cell.splitlines()[0] + "\n\n" + code
        self.backup = cell
        ip.set_next_input(code.rstrip() + "\n", replace=True)
        print("✏️ Your solution cell now holds the corrected version: rerun it, then the sample check.\n"
              "To go back, put undo() as the last line of this cell and run it. If the cell did not change, "
              "your editor does not support this: copy the code from the answer above.")

    def undo(self):
        ip = _ipython()
        current = ip.user_ns.get("In", [""])[-1] if ip else ""
        if not _is_solution_cell(current):
            print(NOT_IN_CELL.format(name="undo"))
        elif self.backup is None:
            print("Nothing to undo: fix() has not changed this cell yet.")
        else:
            ip.set_next_input(self.backup, replace=True)
            self.backup = None
            print("↩️ Your solution cell is back to its version before the last fix(): rerun it.")

    @staticmethod
    def _digest(text):
        return hashlib.sha256(text.encode()).hexdigest()

    def _chat(self, content, cell=None, max_tokens=None):
        """Send one user message; returns the answer, or None if nothing was answered (not counted)."""
        if not self.key:
            try:
                from getpass import getpass
                self.key, self.key_source = getpass("OpenRouter API key: ").strip(), "prompt"
            except Exception:
                pass
            if not self.key:
                print(NO_KEY)
                return None
        user = {"role": "user", "content": content}
        out, errors, notes = _Output(), [], []
        if self._tokens(self.history + [user]) > CONTEXT_WINDOW - MIN_ANSWER_ROOM:  # no room left for an answer
            notes.append(f"⚠️ The conversation reached the context window ({CONTEXT_WINDOW:,} tokens).")
            while self._tokens(self.history + [user]) > CONTEXT_WINDOW - MIN_ANSWER_ROOM:
                pruned = self._prune()
                if pruned is None:
                    out.finish("\n\n".join(notes + ["This message alone is too long for the context window: "
                                                    "shorten your question or your solution cell."]))
                    return None
                notes.append(pruned)
        for model in MODELS:
            while True:  # retried after pruning if the provider says the conversation doesn't fit
                try:
                    for piece in self._stream(model, self.history + [user], max_tokens):
                        out.add(piece)
                    if not out.text.strip():
                        raise _ModelFailed("empty answer")
                except urllib.error.HTTPError as e:
                    message = _error_message(e)
                    if e.code == 401:
                        out.finish(f"🔑 OpenRouter rejected the key ({message}). Check {KEY_NAME}.")
                        return None
                    if e.code == 429 and "per-day" in message:
                        out.finish(f"⏳ Daily limit of free requests reached on your OpenRouter account ({message}).")
                        return None
                    if e.code in (400, 413) and CONTEXT_FULL.search(message) and self._window_full(notes):
                        continue  # same model, shorter conversation
                    errors.append(f"{model}: {e.code} {message}")
                    break
                except (_ModelFailed, urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
                    if out.text:  # the answer broke off midway: don't restart it with another model
                        out.finish(f"⚠️ The answer was interrupted ({e}): ask again.")
                        return None
                    if CONTEXT_FULL.search(str(e)) and self._window_full(notes):
                        continue
                    errors.append(f"{model}: {e}")
                    break
                return self._answered(model, user, out, notes, cell)
        out.finish("\n\n".join(notes + ["⏳ Every model is busy or unavailable right now (free models are often "
                                        "rate-limited). Wait a minute and ask again."])
                   + "\n\n" + "\n".join(f"- `{e}`" for e in errors))
        return None

    def _answered(self, model, user, out, notes, cell):
        """Record a finished answer, prune the conversation if it is over 90% of the window, show the answer."""
        self.history += [user, {"role": "assistant", "content": out.text}]
        if cell is not None:
            self.sent = self._digest(cell)
        self.used = (self.usage or {}).get("total_tokens") or self._tokens(self.history)
        if self.truncated:
            notes.append('✂️ *The answer reached the length limit: `ask("continue")` for the rest.*')
        if self.used >= PRUNE_AT * CONTEXT_WINDOW:
            notes.append(f"⚠️ The conversation uses {self.used:,} of the {CONTEXT_WINDOW:,} tokens of the context "
                         f"window (over {PRUNE_AT:.0%}).")
            pruned = self._prune()
            if pruned:
                notes.append(pruned)
        notes.append(f"<sub>{model.removesuffix(':free')} · context {self.used:,} / {CONTEXT_WINDOW:,} tokens</sub>")
        out.finish("\n\n".join(notes))
        return out.text

    def _window_full(self, notes):
        """The provider rejected the request because the conversation doesn't fit: tell the user and prune.
        True if something was removed (the request can be retried)."""
        notes.append("⚠️ The conversation reached the model's context window.")
        pruned = self._prune()
        if pruned:
            notes.append(pruned)
        return pruned is not None

    def _tokens(self, messages):
        """Token estimate, scaled on the last count reported by OpenRouter (about 4 characters per token before)."""
        per_char = self.used / _chars(self.history) if self.used else 0.25
        return round(per_char * _chars(messages))

    def _room(self, messages):
        """Tokens left in the context window for the answer to these messages."""
        return max(1, min(MAX_TOKENS, CONTEXT_WINDOW - self._tokens(messages)))

    def _prune(self):
        """Drop the oldest question/answer pairs (never the system prompt) to free PRUNE_SHARE of the window.
        Returns a note for the user, or None if there was nothing to drop."""
        target, freed, dropped = PRUNE_SHARE * CONTEXT_WINDOW, 0, 0
        while freed < target and len(self.history) > 2:
            pair = self.history[1:3]
            tokens = self._tokens(pair)
            self.history = self.history[:1] + self.history[3:]
            self.used = max(0, self.used - tokens)  # keeps the tokens-per-character ratio of _tokens right
            freed, dropped = freed + tokens, dropped + len(pair)
        if not dropped:
            return None
        self.sent = None  # the solution cell may have been dropped: send it again with the next question
        return (f"✂️ Context window pruned: the oldest {dropped} messages (about {freed:,} tokens) were removed "
                f"from the conversation.")

    def _stream(self, model, messages, max_tokens=None):
        self.truncated, self.usage = False, None
        body = {"model": model, "messages": messages, "stream": True,
                "max_tokens": max_tokens or self._room(messages),
                "reasoning": {"enabled": False}, "usage": {"include": True}}
        request = urllib.request.Request(API_URL, data=json.dumps(body).encode(), headers={
            "Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/TortueSagace/assisted-coding-challenges",
            "X-Title": "Assisted Coding Challenges"})
        with urllib.request.urlopen(request, timeout=60) as response:
            for raw in response:  # server-sent events: "data: {json}" lines, ": comments", "data: [DONE]"
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                chunk = json.loads(data)
                if "error" in chunk:
                    raise _ModelFailed(chunk["error"].get("message", "provider error"))
                if chunk.get("usage"):  # the last chunk: tokens of the whole request and answer
                    self.usage = chunk["usage"]
                choices = chunk.get("choices") or [{}]
                if choices[0].get("finish_reason") == "length":
                    self.truncated = True
                piece = (choices[0].get("delta") or {}).get("content")
                if piece:
                    yield piece


def _error_message(error):
    try:
        info = json.loads(error.read())["error"]
        raw = (info.get("metadata") or {}).get("raw")
        return str(raw or info.get("message"))[:200]
    except Exception:
        return error.reason


# ------------------------------------------ notebook API ------------------------------------------

def start(context_path, **_ignored):  # **_ignored: older notebooks passed a question budget
    """Set up the assistant for one challenge; context_path is the accN_context.md file of the repo."""
    global _current
    _current = Assistant(Path(context_path).read_text(encoding="utf-8"))
    if _current.key:
        print(f"🤖 Assistant ready (OpenRouter key from {_current.key_source}).")
        print("Write your questions in the cell below.")
    else:
        print(NO_KEY)


def _with_assistant(method, *args):
    if _current is None:
        print("Run the assistant setup cell first.")
        return None
    return getattr(_current, method)(*args)


def ask(question="", *_ignored):
    if not str(question).strip():  # the notebook's placeholder: no question is used up
        print('Write your question between the quotes, e.g. ask("Could a greedy approach work here?"), '
              "then run the cell again.")
        return
    _with_assistant("ask", question)


def fix(hint=""):
    _with_assistant("fix", hint)


def undo():
    _with_assistant("undo")


def reset():
    _with_assistant("reset")
