from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import os
import re
import socket
import time
from collections import deque
from urllib.parse import quote, urlparse

import aiohttp
import chess
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.datastructures import MutableHeaders


APP_VERSION = "1.1.0"
PROMPT_VERSION = "board-fen-legal/v1"
MODEL_RE = re.compile(r"^[A-Za-z0-9._:/@+-]{1,120}$")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MAX_PLIES = 300
MAX_REQUEST_BYTES = 16_384
MAX_PROVIDER_RESPONSE_BYTES = 1_000_000
REQUEST_LIMIT = asyncio.Semaphore(int(os.getenv("CHALLENGE_MAX_CONCURRENT_MOVES", "4")))
MOVE_PATHS = {"/api/move", "/move"}
API_PATHS = MOVE_PATHS | {"/api/health", "/health"}
NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")
CLAUDE_RE = re.compile(r"^claude-(opus|sonnet|haiku|fable|mythos)-(\d+)(?:-(\d{1,2})(?!\d))?")
GEMINI_RE = re.compile(r"^gemini-(\d+)")
THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.IGNORECASE | re.DOTALL)
FENCE_RE = re.compile(r"```(?:json)?", re.IGNORECASE)
THINKING_BUDGETS = {"low": 2048, "medium": 6144, "high": 12288}
GEMINI_BUDGETS = {"low": 1024, "medium": 8192, "high": 24576}
TRUNCATED = "Model ran out of output tokens before giving a move; raise Max output tokens (reasoning uses them too)"
REFUSED = "Model declined to answer this request"


def _csv_env(name: str, default: str = "") -> list[str]:
    return [item.strip().lower() for item in os.getenv(name, default).split(",") if item.strip()]


ALLOWED_PROVIDER_HOSTS = _csv_env("CHALLENGE_ALLOWED_PROVIDER_HOSTS")


class PlayerConfig(BaseModel):
    provider: str = Field(pattern=r"^(openai_compatible|anthropic_compatible|gemini|anthropic)$")
    model: str = Field(min_length=1, max_length=120)
    api_key: str = Field(min_length=1, max_length=512)
    base_url: str | None = Field(default=None, max_length=300)
    reasoning: str = Field(default="default", pattern=r"^(default|low|medium|high)$")
    token_parameter: str = Field(default="auto", pattern=r"^(auto|max_tokens|max_completion_tokens)$")
    max_tokens: int = Field(default=16000, ge=256, le=64000)
    timeout_seconds: int = Field(default=120, ge=10, le=600)

    @field_validator("api_key")
    @classmethod
    def header_safe_api_key(cls, value: str) -> str:
        # Keys become HTTP header values; reject whitespace, controls, and non-ASCII
        # before the HTTP client can raise an exception containing the secret.
        if not all(33 <= ord(char) <= 126 for char in value):
            raise ValueError("API key contains characters unsafe for an HTTP header")
        return value


class MoveRequest(BaseModel):
    player: PlayerConfig
    moves: list[str] = Field(default_factory=list, max_length=MAX_PLIES)


class ProviderFailure(Exception):
    def __init__(self, message: str, status: int = 502):
        self.message = message
        self.status = status


def _is_public_ip(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Reject non-global addresses, including IPv4 embedded in IPv6 transition forms."""
    if not address.is_global:
        return False
    if isinstance(address, ipaddress.IPv6Address):
        embedded = [address.ipv4_mapped, address.sixtofour, address.teredo[1] if address.teredo else None]
        if address in NAT64_PREFIX:
            embedded.append(ipaddress.IPv4Address(int(address) & 0xFFFFFFFF))
        return all(item is None or item.is_global for item in embedded)
    return True


def _host_allowed(host: str) -> bool:
    if not ALLOWED_PROVIDER_HOSTS:
        return True
    host = host.lower().rstrip(".")
    return any(host == rule or (rule.startswith("*.") and host.endswith(rule[1:])) for rule in ALLOWED_PROVIDER_HOSTS)


def _allowed_base_url(raw: str | None, default: str = "https://api.openai.com/v1") -> str:
    url = (raw or default).strip().rstrip("/")
    # urlparse silently removes embedded controls, changing the requested endpoint.
    if not all(33 <= ord(char) <= 126 for char in url) or any(char.isspace() for char in url):
        raise HTTPException(400, "Provider URL must be a public HTTPS endpoint")
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        raise HTTPException(400, "Provider URL must be a public HTTPS endpoint") from None
    try:
        address = ipaddress.ip_address(host) if host else None
    except ValueError:
        address = None
    if (parsed.scheme != "https" or not host
            or parsed.username or parsed.password or port not in (None, 443)
            or parsed.query or parsed.fragment or "//" in parsed.path
            or (address is not None and not _is_public_ip(address))):
        raise HTTPException(400, "Provider URL must be a public HTTPS endpoint")
    if not _host_allowed(host):
        raise HTTPException(400, "Provider host is not allowed on this server")
    return url


class PublicResolver(aiohttp.abc.AbstractResolver):
    async def resolve(self, host: str, port: int = 0, family: int = socket.AF_INET) -> list[dict]:
        try:
            infos = await asyncio.get_running_loop().getaddrinfo(host, port, family=family, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise ProviderFailure("Provider host could not be resolved") from exc
        addresses = []
        for family_value, _, proto, _, sockaddr in infos:
            address = ipaddress.ip_address(sockaddr[0])
            if _is_public_ip(address):
                addresses.append({"hostname": host, "host": sockaddr[0], "port": port,
                                  "family": family_value, "proto": proto, "flags": 0})
        if not addresses:
            raise ProviderFailure("Provider host is not a public address", 400)
        return addresses

    async def close(self) -> None:
        pass


def _board_from_moves(moves: list[str]) -> chess.Board:
    board = chess.Board()
    for uci in moves:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError as exc:
            raise HTTPException(400, "Invalid move history") from exc
        if move not in board.legal_moves:
            raise HTTPException(400, "Invalid move history")
        board.push(move)
    if board.is_game_over(claim_draw=True) or len(moves) >= MAX_PLIES:
        raise HTTPException(400, "Game is over or move limit reached")
    return board


def _prompt(board: chess.Board, retry: bool) -> str:
    legal = ", ".join(f"{move.uci()} ({board.san(move)})" for move in board.legal_moves)
    history = " ".join(move.uci() for move in board.move_stack) or "none"
    san_history = chess.Board().variation_san(board.move_stack) if board.move_stack else "none"
    board_rows = [
        f"{rank}  " + " ".join(
            board.piece_at(chess.square(file, rank - 1)).symbol()
            if board.piece_at(chess.square(file, rank - 1)) else "."
            for file in range(8)
        )
        for rank in range(8, 0, -1)
    ]
    diagram = "\n".join(["    a b c d e f g h", *board_rows, "    a b c d e f g h"])
    correction = "Your previous reply was invalid. Choose exactly one legal move. " if retry else ""
    return (
        "You are playing competitive chess. Choose the strongest move for the side to move. "
        "Give a brief public explanation of the move, not private chain-of-thought. "
        f"{correction}Reply with only JSON: {{\"move\":\"e2e4\",\"explanation\":\"brief reason\"}}.\n"
        "Board orientation: white pieces start on ranks 1-2; files always run a through h left to right. "
        "Uppercase pieces are White, lowercase pieces are Black, and . is empty.\n"
        f"Side to move: {'White' if board.turn else 'Black'}\nBoard:\n{diagram}\n"
        f"FEN: {board.fen()}\nMoves so far (UCI): {history}\n"
        f"Moves so far (SAN): {san_history}\nLegal moves (UCI with SAN): {legal}"
    )


def _answer_object(text: str) -> dict | None:
    """Last JSON object with a "move" field, tolerating prose, code fences and <think> blocks."""
    cleaned = FENCE_RE.sub("", THINK_RE.sub("", text)).strip()
    decoder = json.JSONDecoder()
    found = None
    for start in (index for index, char in enumerate(cleaned) if char == "{"):
        try:
            value, _ = decoder.raw_decode(cleaned, start)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "move" in value:
            found = value
    return found


def _move_from_text(board: chess.Board, move_text: str) -> chess.Move | None:
    text = move_text.strip()
    try:
        move = chess.Move.from_uci(text.lower())
        return move if move in board.legal_moves else None
    except ValueError:
        pass
    try:
        return board.parse_san(text)  # the same leniency applies to both players
    except ValueError:
        return None


def _parse_move(board: chess.Board, text: str) -> tuple[chess.Move, str]:
    data = _answer_object(text)
    move_text = data.get("move") if data else None
    explanation = data.get("explanation", "") if data else ""
    if not isinstance(move_text, str) or not isinstance(explanation, str):
        raise ProviderFailure("Model did not return a valid move")
    move = _move_from_text(board, move_text)
    if move is None:
        raise ProviderFailure("Model returned an illegal move")
    return move, CONTROL_RE.sub("", explanation).strip()[:500]


def _claude_version(model: str) -> tuple[str, float] | None:
    match = CLAUDE_RE.match(model.lower())
    if not match:
        return None
    return match.group(1), float(f"{match.group(2)}.{match.group(3) or 0}")


def _anthropic_thinking(player: PlayerConfig) -> dict:
    """Map the shared reasoning level onto what each Messages-API model accepts."""
    if player.reasoning == "default":
        return {}
    version = _claude_version(player.model)
    modern = version is not None and version[0] != "haiku" and version[1] >= 4.6
    if player.provider == "anthropic" and version is None:
        modern = True  # unrecognized official ID: assume a current model
    if modern:
        return {"thinking": {"type": "adaptive"}, "output_config": {"effort": player.reasoning}}
    # Older Claude models and third-party compatible APIs take a fixed budget below max_tokens.
    budget = min(THINKING_BUDGETS[player.reasoning], player.max_tokens - 1024)
    if budget < 1024:
        raise HTTPException(400, "Raise Max output tokens to at least 2048 to use reasoning with this model")
    return {"thinking": {"type": "enabled", "budget_tokens": budget}}


def _gemini_thinking(player: PlayerConfig) -> dict:
    if player.reasoning == "default":
        return {}
    match = GEMINI_RE.match(player.model.lower())
    if match and int(match.group(1)) >= 3:
        return {"thinkingLevel": player.reasoning.upper()}
    return {"thinkingBudget": min(GEMINI_BUDGETS[player.reasoning], max(128, player.max_tokens - 512))}


def _request_details(player: PlayerConfig, board: chess.Board, retry: bool) -> tuple[str, dict, dict]:
    if not MODEL_RE.fullmatch(player.model):
        raise HTTPException(400, "Invalid model name")
    prompt = _prompt(board, retry)
    if player.provider == "openai_compatible":
        base = _allowed_base_url(player.base_url)
        host = urlparse(base).hostname
        token_field = player.token_parameter if player.token_parameter != "auto" else (
            "max_completion_tokens" if host in {"api.openai.com", "openrouter.ai"} else "max_tokens"
        )
        body = {
            "model": player.model,
            "messages": [{"role": "user", "content": prompt}],
            token_field: player.max_tokens,
        }
        if player.reasoning != "default":
            body["reasoning_effort"] = player.reasoning
        return f"{base}/chat/completions", {"Authorization": f"Bearer {player.api_key}"}, body
    if player.provider == "gemini":
        _allowed_base_url("https://generativelanguage.googleapis.com")
        config = {"maxOutputTokens": player.max_tokens, "responseMimeType": "application/json"}
        thinking = _gemini_thinking(player)
        if thinking:
            config["thinkingConfig"] = thinking
        return (
            f"https://generativelanguage.googleapis.com/v1beta/models/{quote(player.model, safe='-._')}:generateContent",
            {"x-goog-api-key": player.api_key},
            {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": config},
        )
    body = {
        "model": player.model,
        "max_tokens": player.max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    body.update(_anthropic_thinking(player))
    base = _allowed_base_url(
        player.base_url if player.provider == "anthropic_compatible" else None,
        "https://api.anthropic.com/v1",
    )
    return (
        f"{base}/messages",
        {"x-api-key": player.api_key, "anthropic-version": "2023-06-01"},
        body,
    )


def _extract_answer(provider: str, data: dict) -> tuple[str, dict, str | None]:
    """Return answer text, normalized usage, and "length"/"refusal" when the provider stopped early."""
    try:
        if provider == "openai_compatible":
            choice = data["choices"][0]
            answer = choice["message"].get("content") or ""
            stop = {"length": "length", "content_filter": "refusal"}.get(choice.get("finish_reason"))
            usage = data.get("usage") or {}
        elif provider == "gemini":
            usage = data.get("usageMetadata") or {}
            if not data.get("candidates") and (data.get("promptFeedback") or {}).get("blockReason"):
                return "", {}, "refusal"
            candidate = data["candidates"][0]
            parts = (candidate.get("content") or {}).get("parts") or []
            answer = "".join(part.get("text", "") for part in parts if not part.get("thought"))
            reason = candidate.get("finishReason")
            stop = "length" if reason == "MAX_TOKENS" else "refusal" if reason in {
                "SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"} else None
        else:
            answer = "".join(block.get("text", "") for block in data["content"] if block.get("type") == "text")
            stop = {"max_tokens": "length", "refusal": "refusal"}.get(data.get("stop_reason"))
            usage = data.get("usage") or {}
        if not isinstance(answer, str) or (not answer and not stop):
            raise ValueError("No text")
        if provider == "gemini":
            usage = {
                "input_tokens": usage.get("promptTokenCount"),
                "output_tokens": usage.get("candidatesTokenCount"),
                "reasoning_tokens": usage.get("thoughtsTokenCount"),
            }
        elif provider in ("anthropic", "anthropic_compatible"):
            usage = {"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")}
        else:
            details = usage.get("completion_tokens_details") or {}
            usage = {
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": details.get("reasoning_tokens"),
            }
        return answer, {key: value for key, value in usage.items() if isinstance(value, int) and value >= 0}, stop
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise ProviderFailure("Provider returned no usable text") from exc


def _status_failure(status: int) -> ProviderFailure:
    # Provider error bodies are not relayed: some echo parts of the request or key.
    if status == 429:
        return ProviderFailure("Provider rate limit or quota reached; wait and retry", 429)
    if status in (401, 403):
        return ProviderFailure("Provider rejected the API key or model access", 401)
    if status == 404:
        return ProviderFailure("Provider endpoint or model was not found (HTTP 404); check the base URL and model ID")
    if status in (400, 413, 422):
        return ProviderFailure(
            f"Provider rejected the request (HTTP {status}); check the model ID, reasoning, and token settings")
    if 300 <= status < 400:
        return ProviderFailure(f"Provider redirected the request (HTTP {status}); use the final API base URL")
    if status >= 500:
        return ProviderFailure(f"Provider is unavailable (HTTP {status}); retry shortly", 503)
    return ProviderFailure(f"Provider request failed (HTTP {status})")


async def _ask_provider(session: aiohttp.ClientSession, player: PlayerConfig, board: chess.Board, retry: bool) -> tuple[str, dict, str | None]:
    url, headers, body = _request_details(player, board, retry)
    try:
        async with session.post(url, headers=headers, json=body, allow_redirects=False) as response:
            if not 200 <= response.status < 300:
                raise _status_failure(response.status)
            raw = bytearray()
            async for chunk in response.content.iter_chunked(65536):
                raw.extend(chunk)
                if len(raw) > MAX_PROVIDER_RESPONSE_BYTES:
                    raise ProviderFailure("Provider response exceeded size limit")
            data = json.loads(raw)
    except asyncio.TimeoutError as exc:
        raise ProviderFailure("Provider request timed out; raise the timeout or retry", 504) from exc
    except (aiohttp.ClientError, ValueError) as exc:
        raise ProviderFailure("Provider connection or response failed") from exc
    if not isinstance(data, dict):
        raise ProviderFailure("Provider returned an invalid response")
    return _extract_answer(player.provider, data)


class RateLimiter:
    """Sliding-window limit per client address, kept in process memory."""

    def __init__(self, limit: int, window: float = 60.0, max_clients: int = 10_000):
        self.limit = limit
        self.window = window
        self.max_clients = max_clients
        self.hits: dict[str, deque[float]] = {}

    def retry_after(self, key: str, now: float | None = None) -> int | None:
        if self.limit <= 0:
            return None
        now = time.monotonic() if now is None else now
        if key not in self.hits and len(self.hits) >= self.max_clients:
            cutoff = now - self.window
            self.hits = {k: v for k, v in self.hits.items() if v and v[-1] > cutoff}
            if len(self.hits) >= self.max_clients:
                return math.ceil(self.window)
        hits = self.hits.setdefault(key, deque())
        while hits and hits[0] <= now - self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            return max(1, math.ceil(hits[0] + self.window - now))
        hits.append(now)
        return None


MOVE_RATE_LIMIT = RateLimiter(int(os.getenv("CHALLENGE_RATE_LIMIT_PER_MINUTE", "60")))
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
    "connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'"
)


class GuardMiddleware:
    """Bounds request bodies (including chunked ones), rate limits moves, and sets response headers."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope["path"]

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-Content-Type-Options"] = "nosniff"
                headers["Referrer-Policy"] = "no-referrer"
                headers["X-Frame-Options"] = "DENY"
                headers["Cross-Origin-Opener-Policy"] = "same-origin"
                headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=(), payment=()"
                headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
                if os.getenv("CHALLENGE_HSTS") == "1":
                    headers["Strict-Transport-Security"] = "max-age=31536000"
                if path in API_PATHS or path.startswith("/api/"):
                    headers["Cache-Control"] = "no-store"
                elif path.startswith("/assets/"):
                    headers["Cache-Control"] = "public, max-age=31536000, immutable"
                else:
                    headers["Cache-Control"] = "no-cache"
            await send(message)

        async def reject(status: int, detail: str, extra: dict | None = None):
            response = JSONResponse(status_code=status, content={"detail": detail}, headers=extra)
            await response(scope, receive, send_with_headers)

        if scope["method"] == "POST":
            declared = dict(scope.get("headers") or []).get(b"content-length")
            try:
                too_large = declared is not None and int(declared) > MAX_REQUEST_BYTES
            except ValueError:
                too_large = True
            if too_large:
                await reject(413, "Challenge request is too large")
                return
            body = bytearray()
            more = True
            while more:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                body.extend(message.get("body", b""))
                more = message.get("more_body", False)
                if len(body) > MAX_REQUEST_BYTES:
                    await reject(413, "Challenge request is too large")
                    return
            if path in MOVE_PATHS:
                client = (scope.get("client") or ("unknown", 0))[0]
                wait = MOVE_RATE_LIMIT.retry_after(client)
                if wait is not None:
                    await reject(429, f"Too many move requests; retry in {wait} s", {"Retry-After": str(wait)})
                    return
            replayed = False
            original_receive = receive

            async def replay():
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await original_receive()

            receive = replay
        await self.app(scope, receive, send_with_headers)


async def _cancel_on_disconnect(work, is_disconnected, interval: float = 0.25):
    """Run work, cancelling it when the browser aborts so a paused match frees its provider slot."""
    task = asyncio.ensure_future(work)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=interval)
            if done:
                return task.result()
            if await is_disconnected():
                task.cancel()
                raise ProviderFailure("Client closed the request", 499)
    finally:
        if not task.done():
            task.cancel()


app = FastAPI(title="Chess Model Challenge API", version=APP_VERSION, docs_url=None, redoc_url=None, openapi_url=None)
origins = _csv_env("CHALLENGE_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
app.add_middleware(GuardMiddleware)


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, _error: RequestValidationError) -> JSONResponse:
    # FastAPI's default 422 body can echo invalid input, including an API key.
    return JSONResponse(status_code=422, content={"detail": "Invalid challenge request"})


@app.get("/api/health")
@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "chess-model-challenge", "version": APP_VERSION}


@app.post("/api/move")
@app.post("/move")
async def move(request: MoveRequest, http_request: Request = None) -> dict:
    board = _board_from_moves(request.moves)
    before = board.fen()
    started = time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=request.player.timeout_seconds)
    acquired = False

    async def choose() -> tuple[chess.Move, str, dict, int]:
        connector = aiohttp.TCPConnector(resolver=PublicResolver(), use_dns_cache=False)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=False) as session:
            for attempt, retry in enumerate((False, True), start=1):
                answer, usage, stop = await _ask_provider(session, request.player, board, retry)
                try:
                    chosen, explanation = _parse_move(board, answer)
                    return chosen, explanation, usage, attempt
                except ProviderFailure:
                    # A second request would stop the same way, so these fail the turn at once.
                    if stop == "length":
                        raise ProviderFailure(TRUNCATED, 422) from None
                    if stop == "refusal":
                        raise ProviderFailure(REFUSED) from None
                    if retry:
                        raise
        raise ProviderFailure("Model did not return a valid move")

    try:
        try:
            await asyncio.wait_for(REQUEST_LIMIT.acquire(), timeout=0.1)
            acquired = True
        except asyncio.TimeoutError as exc:
            raise ProviderFailure("Challenge server is busy; retry shortly", 503) from exc
        if http_request is None:
            chosen, explanation, usage, attempts = await choose()
        else:
            chosen, explanation, usage, attempts = await _cancel_on_disconnect(choose(), http_request.is_disconnected)
    except ProviderFailure as exc:
        raise HTTPException(exc.status, exc.message) from exc
    finally:
        if acquired:
            REQUEST_LIMIT.release()
    san = board.san(chosen)
    board.push(chosen)
    outcome = board.outcome(claim_draw=True)
    return {
        "move": chosen.uci(), "san": san, "explanation": explanation,
        "fen_before": before, "fen_after": board.fen(),
        "result": outcome.result() if outcome else None,
        "termination": outcome.termination.name if outcome else None,
        "usage": usage, "elapsed_ms": round((time.perf_counter() - started) * 1000),
        "attempts": attempts, "prompt_version": PROMPT_VERSION,
    }


static_dir = os.getenv("CHALLENGE_STATIC_DIR")
if static_dir:
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="challenge-frontend")
