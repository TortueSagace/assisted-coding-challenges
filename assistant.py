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
import time
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
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3.5-lightning:free",
    "apodex/apodex-1.1-mini:free",
    "poolside/laguna-xs-2.1:free",
    "cohere/north-mini-code:free",  # slow and thinks aloud: last resort
]
MAX_TOKENS = 2000
MAX_TOKENS_FIX = 6000  # fix() rewrites the whole cell: more room than a chat answer

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
    """Live Markdown in a notebook (one display updated as the answer streams), plain text elsewhere."""

    def __init__(self):
        self.text, self.last = "", 0.0
        try:
            from IPython.display import Markdown, display
            if _ipython() is None:
                raise ImportError
            self.markdown = Markdown
            self.handle = display(Markdown("*…*"), display_id=True)
        except ImportError:
            self.handle = None

    def add(self, piece):
        self.text += piece
        if self.handle is None:
            print(piece, end="", flush=True)
        elif time.time() - self.last > 0.2:
            self.handle.update(self.markdown(self.text))
            self.last = time.time()

    def finish(self, note):
        if self.handle is None:
            print(("\n\n" if self.text else "") + note)
        else:
            self.handle.update(self.markdown(self.text + ("\n\n" if self.text else "") + note))


class _ModelFailed(Exception):
    pass


class Assistant:
    def __init__(self, context, questions):
        self.context, self.questions_left = context, questions
        self.key, self.key_source = find_key()
        self.backup = None  # the solution cell before the last fix()
        self.reset()

    def reset(self):
        """Forget the conversation (the question budget is kept)."""
        self.history = [{"role": "system", "content": SYSTEM_PROMPT.format(context=self.context)}]
        self.sent = None  # hash of the solution cell last shown to the model

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
        answer = self._chat("\n\n".join(p for p in parts if p), cell, MAX_TOKENS_FIX)
        if answer is None:
            return
        code = _code_block(answer)
        if code is None:
            self.questions_left += 1
            print("No complete my_solution in the answer: your cell was not changed, and this question was not "
                  "counted. Try fix() again (another model may answer), or ask() about the bug.")
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

    def _chat(self, content, cell=None, max_tokens=MAX_TOKENS):
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
        if self.questions_left <= 0:
            print("No questions left for this challenge: the rest is up to you! 💪")
            return None
        messages = self.history + [{"role": "user", "content": content}]

        out, errors = _Output(), []
        for model in MODELS:
            try:
                for piece in self._stream(model, messages, max_tokens):
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
                errors.append(f"{model}: {e.code} {message}")
                continue
            except (_ModelFailed, urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
                if out.text:  # the answer broke off midway: don't restart it with another model
                    out.finish(f"⚠️ The answer was interrupted ({e}). This question was not counted; ask again.")
                    return None
                errors.append(f"{model}: {e}")
                continue
            self.history = messages + [{"role": "assistant", "content": out.text}]
            if cell is not None:
                self.sent = self._digest(cell)
            self.questions_left -= 1
            left = f"{self.questions_left} question{'s' if self.questions_left != 1 else ''} left"
            out.finish(f"<sub>{model.removesuffix(':free')} · {left}</sub>")
            return out.text
        out.finish("⏳ Every model is busy or unavailable right now (free models are often rate-limited). "
                   "Wait a minute and ask again; this question was not counted.\n\n"
                   + "\n".join(f"- `{e}`" for e in errors))
        return None

    def _stream(self, model, messages, max_tokens=None):
        body = {"model": model, "messages": messages, "stream": True, "max_tokens": max_tokens or MAX_TOKENS,
                "reasoning": {"enabled": False}}
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
                choices = chunk.get("choices") or [{}]
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

def start(context_path, questions=10):
    """Set up the assistant for one challenge; context_path is the accN_context.md file of the repo."""
    global _current
    _current = Assistant(Path(context_path).read_text(encoding="utf-8"), questions)
    if _current.key:
        print(f"🤖 Assistant ready ({questions} questions, OpenRouter key from {_current.key_source}).")
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
