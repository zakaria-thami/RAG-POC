"""
The chat model we talk to, and the log of what we said to it.

We go through OpenRouter (https://openrouter.ai): one API key, one paid
account, and access to models from many vendors (OpenAI, Anthropic, Google,
Meta...). Switching model is a one-line change in .env, nothing here moves.

Both values come from .env (see rag/config.py):
  OPENROUTER_API_KEY  the secret key from your OpenRouter account
  MODEL_NAME          an OpenRouter model id, e.g. "openai/gpt-4.1-mini".
                      The full list is at https://openrouter.ai/models

We fail loudly at import time if either is missing: a server that boots but
cannot answer a single question is harder to debug than one that refuses
to start.

NOTE: every call to ask() costs OpenRouter credits. Nothing in this package
calls the model on its own; only the endpoints in api/ do, when a user asks.
"""
import json
import logging
import re
import time
from pathlib import Path

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openrouter import OpenRouter

from rag import config

# ---------- chat log ----------
# A logger of our own that writes to data/chat.log, so the conversation stays
# separate from uvicorn's HTTP lines. Watch it in a second terminal with:
#   PowerShell:  Get-Content data\chat.log -Wait -Tail 20 -Encoding UTF8
#   Git Bash:    tail -f data/chat.log
# -Encoding UTF8 is required: PowerShell reads as ANSI by default and would
# show the model's em-dashes as garbled characters.

LOG_FILE = Path("data/chat.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

chat_log = logging.getLogger("chat")
chat_log.setLevel(logging.INFO)

# propagate=False keeps these lines OUT of the uvicorn window. Without it the
# message travels up to the root logger and gets printed there as well.
chat_log.propagate = False

# uvicorn --reload re-imports this file. Without this guard we would attach a
# second handler and every line would be written twice.
if not chat_log.handlers:
    # encoding="utf-8" is not optional on Windows: the default (cp1252) cannot
    # write the em-dashes and curly quotes the model produces, and the request
    # would crash with UnicodeEncodeError.
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    chat_log.addHandler(handler)


# ---------- the model ----------
if not config.OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is missing.")
if not config.MODEL_NAME:
    raise RuntimeError(
        "MODEL_NAME is missing. Put an OpenRouter model id in .env, "
        "e.g. MODEL_NAME='openai/gpt-4.1-mini' (see https://openrouter.ai/models)"
    )

llm = OpenRouter(
    model=config.MODEL_NAME,
    api_key=config.OPENROUTER_API_KEY,
    max_tokens=config.LLM_MAX_TOKENS,
)

# Written once at startup so chat.log always shows which model produced the
# answers that follow.
chat_log.info("Model ready: %s (via OpenRouter)", config.MODEL_NAME)


def build_messages(system_prompt: str, user_message: str) -> list[ChatMessage]:
    """Turn a (system, user) pair into the message list the model expects.

    A chat model is not given a single string, but a list of messages that each
    carry a role. SYSTEM is the standing instruction, USER is what was actually
    asked. The order matters: system first, then the question.
    """
    messages = []
    # Skip the system message entirely while the prompt is empty - an empty
    # instruction is not the same as no instruction, and some models dislike it.
    if system_prompt.strip():
        messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_prompt))
    messages.append(ChatMessage(role=MessageRole.USER, content=user_message))
    return messages


def token_summary(response) -> str:
    """Turn a reply's token counts into one printable string.

    response.raw is the untouched reply from OpenRouter. It follows the OpenAI
    format, so the counts live in raw.usage. If anything is missing we return
    "tokens n/a" rather than crash a request that already produced a good answer.
    """
    try:
        usage = response.raw.usage
        return (f"{usage.total_tokens} tokens "
                f"({usage.prompt_tokens} in / {usage.completion_tokens} out)")
    except Exception:
        return "tokens n/a"


async def ask(system_prompt: str, user_message: str, label: str = "Q") -> dict:
    """Send one (system, user) pair to the model and return what came back.

    Returns {"text": str, "seconds": float, "tokens": str}. Everything is also
    written to chat.log with `label` so the log shows which pipeline stage
    (routing, answering, plain chat) produced each exchange.

    We call achat() (the async twin of chat()) because our routes are
    "async def": while we wait for OpenRouter to answer, FastAPI can keep
    serving other requests instead of freezing.

    Errors (bad key, no internet, unknown model id, out of credit) are NOT
    swallowed: they are logged and re-raised so FastAPI returns a 500 and the
    traceback stays visible in the uvicorn window.
    """
    chat_log.info("%s: %s", label, user_message)
    started = time.perf_counter()
    try:
        response = await llm.achat(build_messages(system_prompt, user_message))
    except Exception as error:
        chat_log.error("FAILED after %.1fs | %s: %s",
                       time.perf_counter() - started, type(error).__name__, error)
        chat_log.info("-" * 60)
        raise

    seconds = time.perf_counter() - started
    text = response.message.content
    tokens = token_summary(response)

    chat_log.info("A: %s", text)
    chat_log.info("[%s | %.1fs | %s]", config.MODEL_NAME, seconds, tokens)
    chat_log.info("-" * 60)
    return {"text": text, "seconds": seconds, "tokens": tokens}


def parse_json_reply(text: str) -> dict:
    """Read the JSON object out of a model reply.

    Our prompts demand "JSON only", but models still sometimes wrap it in a
    ```json fence or add a sentence before it. We strip the fence and take the
    outermost {...} block. If there is no valid JSON at all we raise, because
    a silent fallback would hide a prompt problem.
    """
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in model reply: {text[:200]!r}")
    return json.loads(cleaned[start:end + 1])
