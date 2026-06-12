"""Anthropic-to-OpenAI proxy for OpenCode Go API."""
import argparse, json, os, sys, uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.request
import urllib.error

AUTH_PATH = os.path.expanduser("~/.local/share/opencode/auth.json")
BASE_URL = os.environ.get("OPENCODE_GO_BASE_URL", "https://opencode.ai/zen/go/v1")
MODEL = os.environ.get("OPENCODE_GO_MODEL", "deepseek-v4-pro")
DEBUG = os.environ.get("OPENCODE_PROXY_DEBUG", "0") == "1"


def dbg(*args):
    if DEBUG:
        print("[opencode-proxy]", *args, file=sys.stderr, flush=True)


def load_api_key():
    with open(AUTH_PATH) as f:
        return json.load(f)["opencode-go"]["key"]


def send_event(wfile, event, data):
    wfile.write(f"event: {event}\n".encode("utf-8"))
    wfile.write(f"data: {json.dumps(data)}\n\n".encode("utf-8"))
    wfile.flush()


class HTTPResponse:
    """Tiny wrapper around urllib response."""
    def __init__(self, status, body=None, response=None):
        self.status = status
        self._body = body
        self._response = response

    def read(self):
        if self._body is not None:
            return self._body
        if self._response is not None:
            return self._response.read()
        return b""

    def iter_lines(self):
        if self._response is None:
            return
        while True:
            line = self._response.readline()
            if not line:
                break
            yield line.rstrip(b"\n")


def http_request(method, url, headers, body=None, stream=False, timeout=300):
    data = None
    if body is not None:
        if isinstance(body, (dict, list)):
            data = json.dumps(body).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")
        elif isinstance(body, bytes):
            data = body
        else:
            data = str(body).encode("utf-8")

    headers.setdefault("User-Agent", "oc-cc/0.1.0")
    headers.setdefault("Accept", "application/json, text/event-stream")

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        if stream:
            return HTTPResponse(resp.status, response=resp)
        return HTTPResponse(resp.status, body=resp.read())
    except urllib.error.HTTPError as e:
        return HTTPResponse(e.code, body=e.read())


def normalize_messages(anthropic_messages):
    """Convert Anthropic messages to OpenAI format."""
    openai_msgs = []
    for m in anthropic_messages:
        role = m.get("role")
        content = m.get("content")
        if role == "tool":
            openai_msgs.append(m)
            continue
        if isinstance(content, str):
            openai_msgs.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            openai_msgs.append({"role": role, "content": str(content)})
            continue

        text_parts = []
        tool_calls = []
        for block in content:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                inp = block.get("input", {})
                if isinstance(inp, dict):
                    inp = json.dumps(inp)
                tool_calls.append({
                    "id": block.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": inp,
                    },
                })
            elif btype == "tool_result":
                tool_content = block.get("content")
                if isinstance(tool_content, list):
                    tool_content = "\n\n".join(str(c.get("text", c)) for c in tool_content)
                elif isinstance(tool_content, dict):
                    tool_content = json.dumps(tool_content)
                openai_msgs.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", block.get("toolCallId", "")),
                    "content": tool_content or "",
                })

        if tool_calls:
            openai_msgs.append({"role": role, "content": None, "tool_calls": tool_calls})
        elif text_parts:
            openai_msgs.append({"role": role, "content": "\n".join(text_parts)})
    return openai_msgs


def normalize_tools(anthropic_tools):
    """Convert Anthropic tool definitions to OpenAI format."""
    openai_tools = []
    for t in anthropic_tools or []:
        schema = t.get("input_schema") or t.get("parameters") or {"type": "object", "properties": {}}
        openai_tools.append({
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": schema,
            },
        })
    return openai_tools


def make_openai_body(anthropic_body):
    messages = []
    system = anthropic_body.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text = "\n\n".join(str(s.get("text", s)) for s in system)
            messages.append({"role": "system", "content": text})
    messages.extend(normalize_messages(anthropic_body.get("messages", [])))

    body = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "max_tokens": anthropic_body.get("max_tokens", 4096),
    }
    tools = normalize_tools(anthropic_body.get("tools"))
    if tools:
        body["tools"] = tools
        tc = anthropic_body.get("tool_choice")
        if tc:
            if tc == "any":
                body["tool_choice"] = "required"
            elif tc in ("auto", "none"):
                body["tool_choice"] = tc
            elif isinstance(tc, dict):
                body["tool_choice"] = {"type": "function", "function": {"name": tc.get("name", "")}}
    return body


def translate_events(source_iter):
    """Yield Anthropic SSE events from OpenAI SSE chunks."""
    text_started = False
    tool_calls = {}
    stop_reason = "end_turn"
    usage = {}
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    started = False

    def next_block_index():
        return 1 if text_started else 0

    for raw in source_iter:
        if not raw:
            continue
        line = raw.decode("utf-8").strip()
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except Exception:
            continue

        choice = (chunk.get("choices") or [None])[0]
        if not choice:
            usage = chunk.get("usage", usage)
            continue

        if not started:
            yield "message_start", {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": MODEL,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            }
            yield "ping", {"type": "ping"}
            started = True

        delta = choice.get("delta") or {}
        content = delta.get("content")
        tool_deltas = delta.get("tool_calls")
        finish = choice.get("finish_reason")

        if content:
            if not text_started:
                yield "content_block_start", {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }
                text_started = True
            yield "content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": content},
            }

        if tool_deltas:
            for td in tool_deltas:
                idx = td.get("index", 0)
                if idx not in tool_calls:
                    tool_calls[idx] = {
                        "id": td.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                        "name": td.get("function", {}).get("name", ""),
                        "args": "",
                    }
                    yield "content_block_start", {
                        "type": "content_block_start",
                        "index": next_block_index() + idx,
                        "content_block": {
                            "type": "tool_use",
                            "id": tool_calls[idx]["id"],
                            "name": tool_calls[idx]["name"],
                            "input": {},
                        },
                    }
                arg_chunk = td.get("function", {}).get("arguments")
                if arg_chunk:
                    tool_calls[idx]["args"] += arg_chunk
                    yield "content_block_delta", {
                        "type": "content_block_delta",
                        "index": next_block_index() + idx,
                        "delta": {"type": "input_json_delta", "partial_json": arg_chunk},
                    }

        if finish:
            if finish == "tool_calls":
                stop_reason = "tool_use"
            elif finish == "length":
                stop_reason = "max_tokens"
            else:
                stop_reason = "end_turn"

    if not started:
        yield "message_start", {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": MODEL,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }

    if text_started:
        yield "content_block_stop", {"type": "content_block_stop", "index": 0}
    for idx in sorted(tool_calls.keys()):
        yield "content_block_stop", {"type": "content_block_stop", "index": next_block_index() + idx}

    yield "message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": usage or {"input_tokens": 0, "output_tokens": 0},
    }
    yield "message_stop", {"type": "message_stop"}


def aggregate_response(r):
    """Convert OpenAI stream to Anthropic non-stream response."""
    text_parts = []
    tool_calls = {}
    stop_reason = "end_turn"
    usage = {}
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    for raw in r.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8").strip()
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except Exception:
            continue
        choice = (chunk.get("choices") or [None])[0]
        if not choice:
            usage = chunk.get("usage", usage)
            continue
        delta = choice.get("delta") or {}
        content = delta.get("content")
        tool_deltas = delta.get("tool_calls")
        finish = choice.get("finish_reason")
        if content:
            text_parts.append(content)
        if tool_deltas:
            for td in tool_deltas:
                idx = td.get("index", 0)
                if idx not in tool_calls:
                    tool_calls[idx] = {"id": td.get("id", ""), "name": td.get("function", {}).get("name", ""), "args": ""}
                arg = td.get("function", {}).get("arguments", "")
                if arg:
                    tool_calls[idx]["args"] += arg
        if finish:
            stop_reason = "tool_use" if finish == "tool_calls" else ("max_tokens" if finish == "length" else "end_turn")

    content = []
    text = "".join(text_parts).strip()
    if text:
        content.append({"type": "text", "text": text})
    for idx in sorted(tool_calls.keys()):
        tc = tool_calls[idx]
        try:
            inp = json.loads(tc["args"])
        except Exception:
            inp = tc["args"]
        content.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": inp})

    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage or {"input_tokens": 0, "output_tokens": 0},
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def handle_error(self, request, client_address):
        pass

    def send_json(self, status, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.end_headers()

    def do_GET(self):
        if self.path == "/v1/models":
            try:
                r = http_request("GET", f"{BASE_URL}/models", headers={
                    "Authorization": f"Bearer {load_api_key()}",
                }, timeout=30)
                self.send_response(r.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(r.read())
            except Exception as e:
                self.send_json(500, {"error": str(e)})
            return
        if self.path in ("/", "/health"):
            self.send_json(200, {"status": "ok", "model": MODEL, "base_url": BASE_URL})
            return
        self.send_json(404, {"error": "not found", "path": self.path})

    def do_POST(self):
        if self.path != "/v1/messages":
            self.send_json(404, {"error": "not found", "path": self.path})
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw)
        except Exception as e:
            self.send_json(400, {"error": f"bad json: {e}"})
            return

        dbg("incoming model:", body.get("model"), "messages:", len(body.get("messages", [])))
        openai_body = make_openai_body(body)
        stream = bool(body.get("stream", True))

        try:
            r = http_request(
                "POST",
                f"{BASE_URL}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {load_api_key()}",
                },
                body=openai_body,
                stream=True,
                timeout=300,
            )
        except Exception as e:
            self.send_json(502, {"error": f"upstream request failed: {e}"})
            return

        if r.status >= 400:
            try:
                err = json.loads(r.read())
            except Exception:
                err = {"raw": r.read().decode("utf-8", errors="replace")}
            self.send_json(r.status, {"error": "upstream error", "details": err})
            return

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            for event, data in translate_events(r.iter_lines()):
                send_event(self.wfile, event, data)
            self.wfile.flush()
            return
        else:
            payload = aggregate_response(r)
            self.send_json(200, payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9877)
    args = parser.parse_args()
    server = HTTPServer((args.host, args.port), Handler)
    print(f"opencode-proxy listening on http://{args.host}:{args.port}", file=sys.stderr)
    print(f"model override: {MODEL}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", file=sys.stderr)


if __name__ == "__main__":
    main()
