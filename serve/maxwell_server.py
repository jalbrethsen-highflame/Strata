"""serve/maxwell_server.py - an OpenAI-compatible server for strata-mx (Maxwell-V1-Flash).

Strata's serve/server.py is built around the Qwen token ids; Maxwell has its own vocabulary, chat template and
special tokens, so this small server drives `strata-mx --serve` over the same engine line protocol (GEN / T / DONE
/ STOP / QUIT):

  * the vendor's tokenizer.json (HF `tokenizers`) and chat template (Jinja), tools and `enable_thinking` included;
  * POST /v1/chat/completions (streaming or not): `reasoning_content` from the <think> block, `tool_calls` from the
    model's <tool_call><function=...><parameter=...> blocks, `usage` with the reused prompt tokens;
  * GET /v1/models, GET /health; an optional API key (Authorization: Bearer ...);
  * the engine reuses the KV of the longest common prefix of the held conversation; the server puts back the
    reasoning of its own earlier answers when a client leaves it out of the history, so the rendered prompt stays
    the token prefix the engine holds;
  * SIGTERM / SIGINT: the engine is asked to QUIT, which writes its KV file (--kv-file), so a restarted server
    continues the conversation without reading it again.

    python serve/maxwell_server.py --engine build/strata-mx --tokenizer DIR --port 18093 [--api-key-file F]
        -- <strata-mx arguments: --model ... --mtp ... --kv-file ...>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import jinja2
from tokenizers import Tokenizer

THINK_END = 200007        # </think>
TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"
DEFAULTS = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}


class Engine:
    """strata-mx --serve as a child process; one request at a time."""

    def __init__(self, exe: str, args: list[str], log: str):
        self.lock = threading.Lock()
        self.log = open(log, "a", buffering=1)
        self.proc = subprocess.Popen([exe, *args, "--serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.log, text=True, bufsize=1)
        self.info = {}
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"strata-mx exited before READY (see {log})")
            if line.startswith("INFO "):
                for kv in line.split()[1:]:
                    k, _, v = kv.partition("=")
                    self.info[k] = v
            if line.startswith("READY"):
                self.ctx = int(line.split()[1])
                break

    def send(self, line: str) -> None:
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def readline(self) -> str:
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("strata-mx exited")
        return line.rstrip("\n")

    def quit(self, timeout: float = 180.0) -> None:
        """QUIT: the engine writes its KV file and exits."""
        try:
            with self.lock:
                self.send("QUIT")
        except (OSError, ValueError):
            pass
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()


class ReasoningMemory:
    """The reasoning of the server's own answers, by (content, tool calls): put back when a client drops it."""

    def __init__(self, n: int = 512, path: str | None = None):
        self.n, self.d, self.path = n, OrderedDict(), path
        if path and os.path.exists(path):   # it survives a restart, as the engine's KV file does
            try:
                self.d = OrderedDict(json.load(open(path)))
            except (OSError, ValueError):
                self.d = OrderedDict()

    def save(self) -> None:
        if not self.path:
            return
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(list(self.d.items()), f)
        os.replace(tmp, self.path)

    @staticmethod
    def key(content: str, tool_calls) -> str:
        tc = json.dumps([[c.get("function", {}).get("name"), c.get("function", {}).get("arguments")] for c in tool_calls or []],
                        sort_keys=True)
        return hashlib.sha256(((content or "").strip() + "\x00" + tc).encode()).hexdigest()

    def put(self, content, tool_calls, reasoning: str) -> None:
        k = self.key(content, tool_calls)
        self.d[k] = reasoning
        self.d.move_to_end(k)
        while len(self.d) > self.n:
            self.d.popitem(last=False)
        self.save()

    def get(self, content, tool_calls):
        return self.d.get(self.key(content, tool_calls))


def parse_tool_calls(text: str, tools) -> list[dict]:
    """<tool_call>\\n<function=NAME>\\n<parameter=P>\\nVALUE\\n</parameter>...</function>\\n</tool_call> -> OpenAI calls."""
    schemas = {}
    for t in tools or []:
        f = t.get("function", t)
        schemas[f.get("name")] = (f.get("parameters") or {}).get("properties") or {}
    calls = []
    for block in re.findall(r"<tool_call>(.*?)(?:</tool_call>|$)", text, re.S):
        m = re.search(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", block, re.S)
        if not m:
            continue
        name, body = m.group(1).strip(), m.group(2)
        args = {}
        for pm in re.finditer(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", body, re.S):
            pname, val = pm.group(1).strip(), pm.group(2)
            ptype = (schemas.get(name, {}).get(pname) or {}).get("type")
            if ptype != "string":
                try:
                    val = json.loads(val)
                except (ValueError, TypeError):
                    pass
            args[pname] = val
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    return calls


class Server:
    def __init__(self, a, engine_args: list[str]):
        self.a = a
        self.tok = Tokenizer.from_file(os.path.join(a.tokenizer, "tokenizer.json"))
        env = jinja2.Environment(loader=jinja2.BaseLoader(), trim_blocks=False, lstrip_blocks=False)

        def raise_exception(msg):
            raise jinja2.TemplateError(msg)

        env.globals["raise_exception"] = raise_exception
        env.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=False, **{k: x for k, x in kw.items() if k == "indent"})
        self.tpl = env.from_string(open(os.path.join(a.tokenizer, "chat_template.jinja")).read())
        self.mem = ReasoningMemory(path=a.reasoning_file)
        self.stopping = False   # SIGTERM: the running request is stopped, the engine quits (and saves its KV)
        self.key = open(a.api_key_file).read().strip() if a.api_key_file else None
        self.engine = Engine(a.engine, engine_args, a.log)
        print(f"[maxwell] engine ready: context {self.engine.ctx}, {self.engine.info}", flush=True)

    # ---- the prompt
    def render(self, req: dict) -> tuple[list[int], bool]:
        msgs = []
        for m in req.get("messages", []):
            m = dict(m)
            if isinstance(m.get("content"), list):   # OpenAI content parts: text only
                m["content"] = "".join(p.get("text", "") for p in m["content"] if isinstance(p, dict))
            if m.get("role") == "assistant":
                tcs = m.get("tool_calls") or []
                if not m.get("reasoning_content"):
                    r = self.mem.get(m.get("content"), tcs)
                    if r:
                        m["reasoning_content"] = r
                fixed = []
                for c in tcs:   # the template iterates the arguments as a mapping
                    c = json.loads(json.dumps(c))
                    f = c.get("function", {})
                    if isinstance(f.get("arguments"), str):
                        try:
                            f["arguments"] = json.loads(f["arguments"] or "{}")
                        except ValueError:
                            f["arguments"] = {}
                    fixed.append(c)
                m["tool_calls"] = fixed
            msgs.append(m)
        kw = dict(req.get("chat_template_kwargs") or {})
        think = kw.get("enable_thinking", True)
        if req.get("reasoning_effort") in ("none", "minimal") or req.get("think") is False:
            think = False
        kw["enable_thinking"] = bool(think)
        text = self.tpl.render(messages=msgs, tools=req.get("tools"), add_generation_prompt=True, **kw)
        return self.tok.encode(text, add_special_tokens=False).ids, bool(think)

    def gen_line(self, req: dict, ids: list[int], max_new: int) -> str:
        keys = []
        t = req.get("temperature", DEFAULTS["temperature"])
        if t is not None and float(t) > 0:
            keys.append(f"temperature={float(t)}")
            keys.append(f"top_p={float(req.get('top_p', DEFAULTS['top_p']))}")
            keys.append(f"top_k={int(req.get('top_k', DEFAULTS['top_k']))}")
            if req.get("min_p"):
                keys.append(f"min_p={float(req['min_p'])}")
        if req.get("presence_penalty"):
            keys.append(f"penalty_present={float(req['presence_penalty'])}")
        if req.get("frequency_penalty"):
            keys.append(f"penalty_freq={float(req['frequency_penalty'])}")
        if req.get("repetition_penalty"):
            keys.append(f"penalty_repeat={float(req['repetition_penalty'])}")
        if req.get("seed") is not None:
            keys.append(f"seed={int(req['seed'])}")
        return f"GEN {max_new} {' '.join(keys)} {','.join(map(str, ids))}"

    # ---- one request: yields ("reasoning" | "content", text) pieces, then ("done", info)
    def run(self, req: dict, cancelled):
        ids, think = self.render(req)
        ctx = self.engine.ctx
        max_new = int(req.get("max_completion_tokens") or req.get("max_tokens") or 0) or (ctx - len(ids))
        max_new = max(1, min(max_new, ctx - len(ids)))
        if len(ids) + 1 > ctx:
            raise ValueError(f"the prompt ({len(ids)} tokens) does not fit the context ({ctx})")
        stops = req.get("stop") or []
        stops = [stops] if isinstance(stops, str) else list(stops)
        with self.engine.lock:
            self.engine.send(self.gen_line(req, ids, max_new))
            r_ids, c_ids = [], []
            in_reason = think
            sent_r = sent_c = 0
            stopped = False
            while True:
                line = self.engine.readline()
                if line.startswith("ERR"):
                    raise ValueError(line[4:])
                if line.startswith("PP "):
                    continue
                if line.startswith("DONE"):
                    f = line.split()
                    info = {"completion_tokens": int(f[1]), "prompt_tokens": int(f[2]), "prompt_ms": float(f[3]),
                            "decode_ms": float(f[4]), "finish": f[5], "accepted": int(f[6]), "offered": int(f[7]),
                            "cached": int(f[8])}
                    break
                if not line.startswith("T ") or stopped:
                    continue
                t = int(line[2:])
                if t in (200001, 2):
                    continue
                if in_reason and t == THINK_END:
                    in_reason = False
                    continue
                (r_ids if in_reason else c_ids).append(t)
                r_text = self.tok.decode(r_ids, skip_special_tokens=False)
                c_text = self.tok.decode(c_ids, skip_special_tokens=False).lstrip("\n")
                for sstr in stops:
                    k = c_text.find(sstr)
                    if k >= 0:
                        c_text = c_text[:k]
                        stopped = True
                if stopped or cancelled() or self.stopping:
                    stopped = True
                    self.engine.send("STOP")
                if not r_text.endswith("�") and len(r_text) > sent_r:
                    yield "reasoning", r_text[sent_r:]
                    sent_r = len(r_text)
                vis = c_text.split(TOOL_OPEN)[0]
                if not stopped and vis.endswith("�"):
                    vis = vis[:-1]
                if len(vis) > sent_c:
                    yield "content", vis[sent_c:]
                    sent_c = len(vis)
        reasoning = self.tok.decode(r_ids, skip_special_tokens=False).strip()
        content_all = self.tok.decode(c_ids, skip_special_tokens=False).lstrip("\n")
        for sstr in stops:
            k = content_all.find(sstr)
            if k >= 0:
                content_all = content_all[:k]
        calls = parse_tool_calls(content_all, req.get("tools")) if TOOL_OPEN in content_all else []
        content = content_all.split(TOOL_OPEN)[0].rstrip() if calls else content_all
        if reasoning:
            self.mem.put(content, calls, reasoning)
        info.update(content=content, reasoning=reasoning, tool_calls=calls)
        if calls:
            info["finish"] = "tool_calls"
        yield "done", info


def make_handler(srv: Server):
    model = srv.a.model_name

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _json(self, code: int, obj) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _auth(self) -> bool:
            if not srv.key:
                return True
            if self.headers.get("Authorization", "") == f"Bearer {srv.key}" or self.headers.get("x-api-key") == srv.key:
                return True
            self._json(401, {"error": {"message": "bad or missing API key", "type": "auth"}})
            return False

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                return self._json(200, {"ok": True, "model": model, "context": srv.engine.ctx})
            if not self._auth():
                return
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                return self._json(200, {"object": "list", "data": [{"id": model, "object": "model", "owned_by": "strata-mx",
                                                                    "context_length": srv.engine.ctx}]})
            self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if not self._auth():
                return
            if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
                return self._json(404, {"error": {"message": "not found"}})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except ValueError:
                return self._json(400, {"error": {"message": "bad JSON"}})
            cid = "chatcmpl-" + uuid.uuid4().hex[:24]
            created = int(time.time())
            gone = {"v": False}
            try:
                if not req.get("stream"):
                    info = None
                    for kind, val in srv.run(req, lambda: False):
                        if kind == "done":
                            info = val
                    msg = {"role": "assistant", "content": info["content"] or (None if info["tool_calls"] else "")}
                    if info["reasoning"]:
                        msg["reasoning_content"] = info["reasoning"]
                    if info["tool_calls"]:
                        msg["tool_calls"] = info["tool_calls"]
                    return self._json(200, {"id": cid, "object": "chat.completion", "created": created, "model": model,
                                            "choices": [{"index": 0, "message": msg, "finish_reason": info["finish"]}],
                                            "usage": usage(info)})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()

                def chunk(delta, finish=None, extra=None):
                    obj = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                           "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                    if extra:
                        obj.update(extra)
                    self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                    self.wfile.flush()

                chunk({"role": "assistant", "content": ""})
                for kind, val in srv.run(req, lambda: gone["v"]):
                    try:
                        if kind == "reasoning":
                            chunk({"reasoning_content": val})
                        elif kind == "content":
                            chunk({"content": val})
                        else:
                            for i, c in enumerate(val["tool_calls"]):
                                chunk({"tool_calls": [{"index": i, **c}]})
                            chunk({}, val["finish"], {"usage": usage(val)})
                    except (BrokenPipeError, ConnectionResetError):
                        gone["v"] = True   # the engine is told to STOP; the request is read to its DONE
                if not gone["v"]:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
            except ValueError as e:
                if not req.get("stream"):
                    self._json(400, {"error": {"message": str(e)}})
            except (BrokenPipeError, ConnectionResetError):
                pass

    return H


def usage(info: dict) -> dict:
    return {"prompt_tokens": info["prompt_tokens"], "completion_tokens": info["completion_tokens"],
            "total_tokens": info["prompt_tokens"] + info["completion_tokens"],
            "prompt_tokens_details": {"cached_tokens": info["cached"]},
            "timings": {"prompt_ms": info["prompt_ms"], "decode_ms": info["decode_ms"],
                        "draft_accepted": info["accepted"], "draft_offered": info["offered"]}}


def main() -> int:
    argv = sys.argv[1:]
    engine_args = []
    if "--" in argv:
        k = argv.index("--")
        argv, engine_args = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--tokenizer", required=True, help="a folder with tokenizer.json and chat_template.jinja")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18093)
    ap.add_argument("--model-name", default="maxwell-v1-flash")
    ap.add_argument("--api-key-file", default=None)
    ap.add_argument("--log", default=os.path.expanduser("~/.local/state/strata-maxwell/engine.log"))
    ap.add_argument("--reasoning-file", default=os.path.expanduser("~/.local/state/strata-maxwell/reasoning.json"),
                    help="the reasoning of earlier answers, kept across restarts (the prompt cache depends on it)")
    a = ap.parse_args(argv)
    os.makedirs(os.path.dirname(a.log), exist_ok=True)
    srv = Server(a, engine_args)
    httpd = ThreadingHTTPServer((a.host, a.port), make_handler(srv))
    httpd.daemon_threads = True

    def stop(*_):
        print("[maxwell] stopping: the engine writes its KV file", flush=True)
        srv.stopping = True
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(f"[maxwell] listening on http://{a.host}:{a.port}/v1 (model {a.model_name})", flush=True)
    httpd.serve_forever()
    srv.engine.quit()
    print("[maxwell] engine stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
