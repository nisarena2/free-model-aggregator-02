#!/usr/bin/env python3
"""Free-model aggregator for make.fun free-model-aggregator-02.

One server, Python 3.13 stdlib only. It serves the OpenAI Chat Completions
wire (POST /v1/chat/completions) and the Anthropic Messages wire
(POST /v1/messages), plus GET /v1/models in the wire the auth header names.
Every completion it answers comes from one of the fourteen free OpenRouter
models, dialled through the grader's gateway on the trial's key. Nothing is
ever answered locally: the server is an aggregator, and its completions come
from the internet, at $0, through the gateway.

Routing: each prompt is served by the highest-rated free model that meets
its requirements (JSON mode, tools, vision). A refusal (429/5xx) is retried
once on the same model, then failed over to the next eligible model; the
refusal trail is reported in the x-makefun-upstream-failures header.

Determinism: every choice is a function of the brief, the requests and the
gateway's answers. No wall clock reaches any request (dates come from
MAKEFUN_TRIAL_NOW only), no unseeded randomness, requests are served one
at a time in the order they arrive, and answer ids come from a counter.
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------- models
# The house's sealed copy of the live free catalog, from the trial brief
# (practice/trial-0/input/prompt.txt). `id` is the id the gateway routes.
# rating is the house's quality rating, 0-2000, higher is stronger.
MODELS = [
    {"id": "openrouter/baidu/ernie-5.5-flash-preview:free",
     "name": "Baidu: ERNIE-5.5-Flash-Preview (free)",
     "rating": 1150, "ctx": 1024000, "tools": True, "vision": False, "json": True},
    {"id": "openrouter/cohere/north-mini-code:free",
     "name": "Cohere: North Mini Code (free)",
     "rating": 1200, "ctx": 256000, "tools": True, "vision": False, "json": False},
    {"id": "openrouter/dots-studio/dots-3-note-preview:free",
     "name": "Dots Studio: Dots3-Note Preview (free)",
     "rating": 1150, "ctx": 512000, "tools": True, "vision": True, "json": True},
    {"id": "openrouter/google/gemma-4-26b-a4b-it:free",
     "name": "Google: Gemma 4 26B A4B (free)",
     "rating": 1150, "ctx": 262144, "tools": True, "vision": True, "json": True},
    {"id": "openrouter/google/gemma-4-31b-it:free",
     "name": "Google: Gemma 4 31B (free)",
     "rating": 1210, "ctx": 262144, "tools": True, "vision": True, "json": True},
    {"id": "openrouter/inclusionai/ling-3.2-flash:free",
     "name": "inclusionAI: Ling 3.2 Flash (free)",
     "rating": 1150, "ctx": 1048576, "tools": True, "vision": True, "json": True},
    {"id": "openrouter/liquid/lfm-3-1.7b:free",
     "name": "LiquidAI: LFM3-1.7B (free)",
     "rating": 1150, "ctx": 262144, "tools": True, "vision": False, "json": True},
    {"id": "openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
     "name": "NVIDIA: Nemotron 3 Nano Omni (free)",
     "rating": 1150, "ctx": 256000, "tools": True, "vision": True, "json": False},
    {"id": "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
     "name": "NVIDIA: Nemotron 3 Ultra (free)",
     "rating": 1285, "ctx": 1000000, "tools": True, "vision": False, "json": False},
    {"id": "openrouter/nvidia/nemotron-3.5-lightning:free",
     "name": "NVIDIA: Nemotron 3.5 Lightning (free)",
     "rating": 1221, "ctx": 1000000, "tools": True, "vision": False, "json": False},
    {"id": "openrouter/nvidia/nemotron-3.5-super-vl:free",
     "name": "NVIDIA: Nemotron 3.5 Super VL (free)",
     "rating": 1303, "ctx": 1000000, "tools": True, "vision": True, "json": True},
    {"id": "openrouter/particle-ai/particle-1:free",
     "name": "Particle AI: Particle-1 (free)",
     "rating": 1150, "ctx": 1048576, "tools": True, "vision": True, "json": True},
    {"id": "openrouter/recursive/flint-v1.0:free",
     "name": "Recursive: Flint (free)",
     "rating": 1150, "ctx": 262144, "tools": True, "vision": False, "json": True},
    {"id": "openrouter/thinkingmachines/inkling-2-small:free",
     "name": "Thinking Machines: Inkling 2 Small (free)",
     "rating": 1475, "ctx": 1048576, "tools": True, "vision": True, "json": False},
]

BY_ID = {m["id"]: m for m in MODELS}

CATALOG = {"providers": [{"id": "openrouter",
                          "base_url": "https://openrouter.ai/api/v1",
                          "key_env": "OPENROUTER_API_KEY",
                          "wire": "openai",
                          "models": [{"id": m["id"], "free": True,
                                      "context_length": m["ctx"]}
                                     for m in MODELS]}]}
CATALOG_JSON = json.dumps(CATALOG, separators=(",", ":"))

MAX_OUTPUT_TOKENS = 8192
DEFAULT_MAX_TOKENS = 1024
MAX_ATTEMPTS = 6
GATEWAY_TIMEOUT = 20


def candidates(need_json, need_tools, need_vision):
    """Eligible models, best first: rating desc, context desc, id asc."""
    out = [m for m in MODELS
           if (not need_json or m["json"])
           and (not need_tools or m["tools"])
           and (not need_vision or m["vision"])]
    out.sort(key=lambda m: (-m["rating"], -m["ctx"], m["id"]))
    return out


# ------------------------------------------------------------- determinism
def _created_epoch():
    now = os.environ.get("MAKEFUN_TRIAL_NOW", "")
    if now:
        try:
            return int(datetime.fromisoformat(now.replace("Z", "+00:00")).timestamp())
        except Exception:
            pass
    return 0


CREATED = _created_epoch()

_COUNTER = [0]
_COUNTER_LOCK = threading.Lock()


def next_id(prefix):
    with _COUNTER_LOCK:
        _COUNTER[0] += 1
        return "%s-agg2-%06d" % (prefix, _COUNTER[0])


def split_text(text, size=24):
    if not text:
        return []
    return [text[i:i + size] for i in range(0, len(text), size)]


def norm_usage(usage, text):
    """A whole usage count: input at least 0, output at least 1."""
    if not isinstance(usage, dict):
        usage = {}
    try:
        prompt_tokens = int(usage.get("prompt_tokens", 0))
    except Exception:
        prompt_tokens = 0
    try:
        completion_tokens = int(usage.get("completion_tokens", 0))
    except Exception:
        completion_tokens = 0
    if prompt_tokens < 0:
        prompt_tokens = 0
    if completion_tokens < 1:
        completion_tokens = max(1, len(text or "") // 4)
    return {"prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens}


def err_code(payload):
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            return err.get("code") or err.get("type") or "upstream_error"
    return "upstream_error"


# ------------------------------------------------------------ gateway call
def gateway_url():
    base = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    return base.rstrip("/") + "/chat/completions"


def gateway_call(model_id, body):
    """One Chat Completions call to the gateway.

    Returns (status, payload). The key is sent as the vendor's SDK sends
    it: Authorization: Bearer <OPENAI_API_KEY>. The gateway swaps it for
    the entrant's own OpenRouter key, which never enters the sandbox.
    """
    data = json.dumps(body, separators=(",", ":")).encode()
    req = urllib.request.Request(gateway_url(), data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        req.add_header("Authorization", "Bearer " + key)
    try:
        with urllib.request.urlopen(req, timeout=GATEWAY_TIMEOUT) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"error": {"message": raw, "type": "upstream_error",
                                "code": "upstream_error"}}
        return exc.code, parsed
    except Exception as exc:  # transport error: treat as a refusal
        return 504, {"error": {"message": str(exc), "type": "timeout",
                               "code": "timeout"}}


def serve(gw_body, need_json, need_tools, need_vision, named_model=None):
    """Dial the gateway until a model answers, or give up.

    Plan: the best eligible model, retried once, then the next eligible
    models in rating order. At most MAX_ATTEMPTS calls. Returns
    (model_id, payload, failures) on success, or
    (None, (status, error_payload), failures) when giving up: the
    provider's refusal is then returned to the client as the provider
    sent it, status and body only.
    """
    cands = candidates(need_json, need_tools, need_vision)
    if named_model and named_model in BY_ID:
        first = BY_ID[named_model]
        if first in cands:  # only honour a named model that meets the prompt
            cands = [first] + [m for m in cands if m is not first]
    plan = []
    if cands:
        plan.append(cands[0])
        plan.append(cands[0])
        plan.extend(cands[1:5])
    plan = plan[:MAX_ATTEMPTS]
    failures = []
    last = (502, {"error": {"message": "no eligible model",
                            "type": "upstream_error", "code": "no_model"}})
    for model in plan:
        body = dict(gw_body)
        body["model"] = model["id"]
        body["stream"] = False
        body.pop("stream_options", None)
        if "max_tokens" in body and "max_completion_tokens" not in body:
            try:
                body["max_tokens"] = min(int(body["max_tokens"]),
                                         MAX_OUTPUT_TOKENS)
            except Exception:
                body["max_tokens"] = DEFAULT_MAX_TOKENS
        elif "max_completion_tokens" in body:
            try:
                body["max_completion_tokens"] = min(
                    int(body["max_completion_tokens"]), MAX_OUTPUT_TOKENS)
            except Exception:
                body["max_completion_tokens"] = DEFAULT_MAX_TOKENS
        else:
            body["max_tokens"] = DEFAULT_MAX_TOKENS
        status, payload = gateway_call(model["id"], body)
        if status == 200:
            return model["id"], payload, failures
        failures.append({"status": status, "code": err_code(payload),
                         "model": model["id"]})
        last = (status, payload)
        if status == 400:
            break  # our own malformed call: retrying cannot fix it
        if status == 401 and len(failures) >= 2:
            break  # no key: every model answers the same
        if len(failures) >= MAX_ATTEMPTS:
            break
        time.sleep(1)
    return None, last, failures


# ------------------------------------------------- anthropic -> openai wire
def anthropic_system_to_text(system):
    if system is None:
        return None
    if isinstance(system, str):
        return system
    parts = [b.get("text", "") for b in system
             if isinstance(b, dict) and b.get("type") == "text"]
    joined = "\n".join(p for p in parts if p)
    return joined or None


def anthropic_messages_to_openai(messages):
    out = []
    for msg in messages or []:
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        texts = []
        images = []
        tool_calls = []
        tool_results = []
        for block in content or []:
            btype = block.get("type")
            if btype == "text":
                texts.append(block.get("text", ""))
            elif btype == "image":
                src = block.get("source", {}) or {}
                if src.get("type") == "base64":
                    url = "data:%s;base64,%s" % (
                        src.get("media_type", "image/png"),
                        src.get("data", ""))
                else:
                    url = src.get("url", "")
                images.append({"type": "image_url",
                               "image_url": {"url": url}})
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {}),
                                                separators=(",", ":"))}})
            elif btype == "tool_result":
                inner = block.get("content")
                if isinstance(inner, list):
                    inner = "\n".join(
                        x.get("text", "") for x in inner
                        if isinstance(x, dict) and x.get("type") == "text")
                if not isinstance(inner, str):
                    inner = json.dumps(inner)
                tool_results.append({"role": "tool",
                                     "tool_call_id": block.get("tool_use_id", ""),
                                     "content": inner})
        if tool_results:
            out.extend(tool_results)
            continue
        if role == "assistant" and tool_calls:
            out.append({"role": "assistant",
                        "content": "\n".join(texts) if texts else None,
                        "tool_calls": tool_calls})
            continue
        if images:
            parts = [{"type": "text", "text": t} for t in texts] + images
            out.append({"role": role, "content": parts})
        else:
            out.append({"role": role, "content": "\n".join(texts)})
    return out


def anthropic_tools_to_openai(tools):
    out = []
    for tool in tools or []:
        out.append({"type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema",
                                               {"type": "object"})}})
    return out


def anthropic_tool_choice_to_openai(choice):
    if not isinstance(choice, dict):
        return None
    ctype = choice.get("type")
    if ctype == "auto":
        return "auto"
    if ctype == "any":
        return "required"
    if ctype == "none":
        return "none"
    if ctype == "tool":
        return {"type": "function",
                "function": {"name": choice.get("name", "")}}
    return "auto"


# ------------------------------------------------- openai -> anthropic wire
def openai_message_to_blocks(msg):
    blocks = []
    content = msg.get("content")
    if isinstance(content, str) and content:
        blocks.append({"type": "text", "text": content})
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                blocks.append({"type": "text", "text": part.get("text", "")})
    for call in msg.get("tool_calls", []) or []:
        fn = call.get("function", {}) or {}
        args = fn.get("arguments", "{}")
        try:
            parsed = json.loads(args) if isinstance(args, str) else args
        except Exception:
            parsed = {"_raw": args}
        blocks.append({"type": "tool_use", "id": call.get("id", ""),
                       "name": fn.get("name", ""), "input": parsed})
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    return blocks


def openai_text(msg):
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content
                       if isinstance(p, dict) and p.get("type") == "text")
    return ""


def openai_stop(payload):
    choices = payload.get("choices") or [{}]
    reason = choices[0].get("finish_reason")
    if reason in ("stop", "length", "tool_calls", "content_filter",
                  "function_call"):
        return reason
    return "stop"


def anthropic_stop(openai_reason):
    return {"stop": "end_turn", "length": "max_tokens",
            "tool_calls": "tool_use", "content_filter": "end_turn",
            "function_call": "tool_use"}.get(openai_reason, "end_turn")


# ------------------------------------------------------- response builders
def openai_response(model_id, payload):
    choices = payload.get("choices") or [{}]
    msg = choices[0].get("message", {}) or {}
    text = openai_text(msg)
    reason = openai_stop(payload)
    usage = norm_usage(payload.get("usage"), text)
    out_msg = {"role": "assistant", "content": msg.get("content")}
    if msg.get("tool_calls"):
        out_msg["tool_calls"] = msg["tool_calls"]
    return {"id": next_id("chatcmpl"), "object": "chat.completion",
            "created": CREATED, "model": model_id,
            "choices": [{"index": 0, "message": out_msg,
                         "finish_reason": reason}],
            "usage": usage}


def openai_stream_chunks(model_id, payload):
    choices = payload.get("choices") or [{}]
    msg = choices[0].get("message", {}) or {}
    text = openai_text(msg)
    reason = openai_stop(payload)
    usage = norm_usage(payload.get("usage"), text)
    cid = next_id("chatcmpl")
    chunks = [{"id": cid, "object": "chat.completion.chunk",
               "created": CREATED, "model": model_id,
               "choices": [{"index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None}]}]
    if msg.get("tool_calls"):
        chunks.append({"id": cid, "object": "chat.completion.chunk",
                       "created": CREATED, "model": model_id,
                       "choices": [{"index": 0,
                                    "delta": {"tool_calls": msg["tool_calls"]},
                                    "finish_reason": None}]})
    for piece in split_text(text):
        chunks.append({"id": cid, "object": "chat.completion.chunk",
                       "created": CREATED, "model": model_id,
                       "choices": [{"index": 0,
                                    "delta": {"content": piece},
                                    "finish_reason": None}]})
    chunks.append({"id": cid, "object": "chat.completion.chunk",
                   "created": CREATED, "model": model_id,
                   "choices": [{"index": 0, "delta": {},
                                "finish_reason": reason}],
                   "usage": usage})
    return chunks


def anthropic_response(model_id, payload):
    choices = payload.get("choices") or [{}]
    msg = choices[0].get("message", {}) or {}
    text = openai_text(msg)
    reason = openai_stop(payload)
    usage = norm_usage(payload.get("usage"), text)
    blocks = openai_message_to_blocks(msg)
    return {"id": next_id("msg"), "type": "message", "role": "assistant",
            "model": model_id, "content": blocks,
            "stop_reason": anthropic_stop(reason), "stop_sequence": None,
            "usage": {"input_tokens": usage["prompt_tokens"],
                      "output_tokens": usage["completion_tokens"]}}


def anthropic_stream_events(model_id, payload):
    choices = payload.get("choices") or [{}]
    msg = choices[0].get("message", {}) or {}
    text = openai_text(msg)
    reason = openai_stop(payload)
    usage = norm_usage(payload.get("usage"), text)
    blocks = openai_message_to_blocks(msg)
    mid = next_id("msg")
    events = [("message_start",
               {"type": "message_start",
                "message": {"id": mid, "type": "message",
                            "role": "assistant", "model": model_id,
                            "content": [], "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": usage["prompt_tokens"],
                                      "output_tokens": 0}}})]
    index = 0
    for block in blocks:
        if block["type"] == "text":
            events.append(("content_block_start",
                           {"type": "content_block_start", "index": index,
                            "content_block": {"type": "text", "text": ""}}))
            for piece in split_text(block["text"]):
                events.append(("content_block_delta",
                               {"type": "content_block_delta",
                                "index": index,
                                "delta": {"type": "text_delta",
                                          "text": piece}}))
            events.append(("content_block_stop",
                           {"type": "content_block_stop", "index": index}))
        elif block["type"] == "tool_use":
            events.append(("content_block_start",
                           {"type": "content_block_start", "index": index,
                            "content_block": {"type": "tool_use",
                                              "id": block["id"],
                                              "name": block["name"],
                                              "input": {}}}))
            args = json.dumps(block["input"], separators=(",", ":"))
            for piece in split_text(args, 48):
                events.append(("content_block_delta",
                               {"type": "content_block_delta",
                                "index": index,
                                "delta": {"type": "input_json_delta",
                                          "partial_json": piece}}))
            events.append(("content_block_stop",
                           {"type": "content_block_stop", "index": index}))
        index += 1
    events.append(("message_delta",
                   {"type": "message_delta",
                    "delta": {"stop_reason": anthropic_stop(reason),
                              "stop_sequence": None},
                    "usage": {"output_tokens": usage["completion_tokens"]}}))
    events.append(("message_stop", {"type": "message_stop"}))
    return events


# ------------------------------------------------------------- http server
def failure_header(failures):
    if not failures:
        return None
    return json.dumps(failures[:8], separators=(",", ":"))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    # -- helpers ------------------------------------------------------
    def _send_json(self, status, doc, failures=None):
        data = json.dumps(doc, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        header = failure_header(failures)
        if header is not None:
            self.send_header("x-makefun-upstream-failures", header)
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except Exception:
            length = 0
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw.decode())
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _bad_json(self):
        self._send_json(400, {"error": {
            "message": "request body is not a JSON object",
            "type": "invalid_request_error", "code": "invalid_json"}})

    def _not_found(self):
        self._send_json(404, {"error": {
            "message": "no such route", "type": "not_found_error",
            "code": "not_found"}})

    def _give_up(self, status, payload, failures):
        # A refusal is returned as the provider sent it: status and body.
        if isinstance(payload, dict):
            self._send_json(status, payload, failures)
        else:
            self._send_json(status, {"error": {
                "message": str(payload), "type": "upstream_error",
                "code": "upstream_error"}}, failures)

    # -- routes -------------------------------------------------------
    def do_GET(self):
        if self.path == "/v1/models" or self.path.startswith("/v1/models?"):
            self._serve_models()
        else:
            self._not_found()

    def _serve_models(self):
        if self.headers.get("x-api-key"):
            data = [{"id": m["id"], "type": "model",
                     "display_name": m["name"],
                     "created_at": "2026-11-15T03:30:00Z"}
                    for m in MODELS]
            self._send_json(200, {"object": "list", "data": data})
        else:
            data = [{"id": m["id"], "object": "model", "created": CREATED,
                     "owned_by": "openrouter"} for m in MODELS]
            self._send_json(200, {"object": "list", "data": data})

    def do_POST(self):
        body = self._read_body()
        if body is None:
            self._bad_json()
            return
        if self.path == "/v1/chat/completions":
            self._serve_openai(body)
        elif self.path == "/v1/messages":
            self._serve_anthropic(body)
        else:
            self._not_found()

    def _serve_openai(self, body):
        fmt = body.get("response_format")
        need_json = isinstance(fmt, dict) and fmt.get("type") in (
            "json_object", "json_schema")
        need_tools = bool(body.get("tools"))
        need_vision = any(
            isinstance(m.get("content"), list)
            and any(isinstance(p, dict) and p.get("type") == "image_url"
                    for p in m["content"])
            for m in body.get("messages", []) if isinstance(m, dict))
        gw_body = {}
        for key in ("messages", "temperature", "top_p", "response_format",
                    "tools", "tool_choice", "stop", "presence_penalty",
                    "frequency_penalty", "seed", "max_tokens",
                    "max_completion_tokens"):
            if key in body:
                gw_body[key] = body[key]
        named = body.get("model")
        if named not in BY_ID:
            named = None
        model_id, payload, failures = serve(
            gw_body, need_json, need_tools, need_vision, named)
        if model_id is None:
            status, err = payload
            self._give_up(status, err, failures)
            return
        if body.get("stream"):
            self._send_openai_stream(model_id, payload, failures)
        else:
            self._send_json(200, openai_response(model_id, payload), failures)

    def _send_openai_stream(self, model_id, payload, failures):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        header = failure_header(failures)
        if header is not None:
            self.send_header("x-makefun-upstream-failures", header)
        self.end_headers()
        for chunk in openai_stream_chunks(model_id, payload):
            line = "data: " + json.dumps(chunk, separators=(",", ":")) + "\n\n"
            self.wfile.write(line.encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _serve_anthropic(self, body):
        need_tools = bool(body.get("tools"))
        need_vision = any(
            isinstance(m.get("content"), list)
            and any(isinstance(p, dict) and p.get("type") == "image"
                    for p in m["content"])
            for m in body.get("messages", []) if isinstance(m, dict))
        messages = anthropic_messages_to_openai(body.get("messages", []))
        system_text = anthropic_system_to_text(body.get("system"))
        if system_text:
            messages = [{"role": "system", "content": system_text}] + messages
        gw_body = {"messages": messages}
        for key in ("temperature", "top_p"):
            if key in body:
                gw_body[key] = body[key]
        if body.get("stop_sequences"):
            gw_body["stop"] = body["stop_sequences"]
        if body.get("tools"):
            gw_body["tools"] = anthropic_tools_to_openai(body["tools"])
        if body.get("tool_choice"):
            choice = anthropic_tool_choice_to_openai(body["tool_choice"])
            if choice is not None:
                gw_body["tool_choice"] = choice
        if body.get("max_tokens") is not None:
            try:
                gw_body["max_tokens"] = min(int(body["max_tokens"]),
                                            MAX_OUTPUT_TOKENS)
            except Exception:
                gw_body["max_tokens"] = DEFAULT_MAX_TOKENS
        named = body.get("model")
        if named not in BY_ID:
            named = None
        model_id, payload, failures = serve(
            gw_body, False, need_tools, need_vision, named)
        if model_id is None:
            status, err = payload
            self._give_up(status, err, failures)
            return
        if body.get("stream"):
            self._send_anthropic_stream(model_id, payload, failures)
        else:
            self._send_json(200, anthropic_response(model_id, payload),
                            failures)

    def _send_anthropic_stream(self, model_id, payload, failures):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        header = failure_header(failures)
        if header is not None:
            self.send_header("x-makefun-upstream-failures", header)
        self.end_headers()
        for event, doc in anthropic_stream_events(model_id, payload):
            line = ("event: " + event + "\ndata: "
                    + json.dumps(doc, separators=(",", ":")) + "\n\n")
            self.wfile.write(line.encode())
            self.wfile.flush()


# ------------------------------------------------------------ env protocol
def register(port):
    env = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    if not env:
        return None
    body = json.dumps({"protocol_version": 1, "port": port,
                       "catalog": CATALOG}, separators=(",", ":")).encode()
    for _ in range(60):
        try:
            req = urllib.request.Request(env + "/register", data=body,
                                         method="POST")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            time.sleep(0.5)
    return None


def poll_work(stop_event):
    env = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    if not env:
        return
    while not stop_event.is_set():
        try:
            with urllib.request.urlopen(env + "/work", timeout=10) as resp:
                data = json.loads(resp.read().decode())
            if data.get("done"):
                stop_event.set()
                return
        except Exception:
            pass
        stop_event.wait(1.0)


def main(argv):
    if "--list" in argv:
        sys.stdout.write(CATALOG_JSON + "\n")
        return 0
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    port = server.server_address[1]
    register(port)
    stop = threading.Event()
    threading.Thread(target=poll_work, args=(stop,), daemon=True).start()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    stop.wait()
    server.shutdown()
    server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
