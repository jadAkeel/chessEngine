from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
import re
from typing import Any
from urllib.parse import quote

import aiohttp
import chess


DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_GEMINI_THINKING_LEVEL = "high"
DEFAULT_GEMINI_TIMEOUT_SECONDS = 30.0
DEFAULT_GEMINI_MAX_ATTEMPTS = 2
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_THINKING_LEVELS = {"low", "medium", "high"}
_REQUEST_SEMAPHORE = asyncio.Semaphore(2)


class GeminiOpponentError(RuntimeError):
    """Base error for the Gemini opponent."""


class GeminiNotConfiguredError(GeminiOpponentError):
    """Raised when the backend has no Gemini API key."""


class GeminiRateLimitError(GeminiOpponentError):
    """Raised when the provider rejects the request for quota reasons."""


class GeminiTimeoutError(GeminiOpponentError):
    """Raised when the provider does not answer in time."""


class GeminiProviderError(GeminiOpponentError):
    """Raised when the provider request fails."""


class GeminiInvalidResponseError(GeminiOpponentError):
    """Raised when Gemini does not return a legal chess move."""


@dataclass(frozen=True)
class GeminiMove:
    move: chess.Move
    model: str
    thinking_level: str


class GeminiOpponent:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        thinking_level: str | None = None,
        timeout_seconds: float | None = None,
        max_attempts: int = DEFAULT_GEMINI_MAX_ATTEMPTS,
    ) -> None:
        self.api_key = (api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")).strip()
        self.model = (model or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip()
        self.thinking_level = (
            thinking_level
            or os.environ.get("GEMINI_THINKING_LEVEL")
            or DEFAULT_GEMINI_THINKING_LEVEL
        ).strip().lower()
        self.timeout_seconds = _resolve_timeout(timeout_seconds)
        self.max_attempts = max(1, min(2, int(max_attempts)))

        if not _MODEL_NAME_RE.fullmatch(self.model):
            raise GeminiNotConfiguredError("Invalid Gemini model name")
        if self.thinking_level not in _THINKING_LEVELS:
            raise GeminiNotConfiguredError("Invalid Gemini thinking level")

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def build_payload(self, board: chess.Board, *, retry: bool = False) -> dict[str, Any]:
        legal_moves = [move.uci() for move in board.legal_moves]
        if not legal_moves:
            raise GeminiInvalidResponseError("The game has no legal moves")

        correction = (
            " Your previous answer was invalid. Return exactly one move from the legal list."
            if retry
            else ""
        )
        prompt = (
            "You are playing a chess game. Choose the strongest legal move for the side to move. "
            "Return only the structured JSON requested by the response schema."
            f"{correction}\n\n"
            f"FEN: {board.fen()}\n"
            f"Side to move: {'white' if board.turn == chess.WHITE else 'black'}\n"
            f"Legal moves (UCI): {', '.join(legal_moves)}"
        )

        return {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": {
                    "type": "object",
                    "properties": {
                        "move": {
                            "type": "string",
                            "description": "The selected legal move in UCI notation.",
                            "enum": legal_moves,
                        }
                    },
                    "required": ["move"],
                    "additionalProperties": False,
                },
                "thinkingConfig": {
                    "thinkingLevel": self.thinking_level.upper(),
                    "includeThoughts": False,
                },
            },
        }

    async def choose_move(self, board: chess.Board) -> GeminiMove:
        if not self.configured:
            raise GeminiNotConfiguredError("Gemini is not configured")
        if board.is_game_over(claim_draw=True):
            raise GeminiInvalidResponseError("The game is already over")

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with _REQUEST_SEMAPHORE:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                for attempt in range(self.max_attempts):
                    response_data = await self._request(session, board, retry=attempt > 0)
                    try:
                        move = parse_gemini_move(board, response_data)
                    except GeminiInvalidResponseError:
                        if attempt + 1 >= self.max_attempts:
                            raise
                        continue
                    return GeminiMove(
                        move=move,
                        model=self.model,
                        thinking_level=self.thinking_level,
                    )

        raise GeminiInvalidResponseError("Gemini did not return a legal move")

    async def _request(
        self,
        session: aiohttp.ClientSession,
        board: chess.Board,
        *,
        retry: bool,
    ) -> dict[str, Any]:
        model_path = quote(self.model, safe="-._")
        url = f"{GEMINI_API_BASE_URL}/models/{model_path}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        try:
            async with session.post(url, headers=headers, json=self.build_payload(board, retry=retry)) as response:
                if response.status == 429:
                    raise GeminiRateLimitError("Gemini rate limit reached")
                if response.status >= 400:
                    raise GeminiProviderError(f"Gemini request failed with status {response.status}")
                try:
                    payload = await response.json(content_type=None)
                except (aiohttp.ContentTypeError, json.JSONDecodeError, ValueError) as exc:
                    raise GeminiInvalidResponseError("Gemini returned malformed JSON") from exc
        except asyncio.TimeoutError as exc:
            raise GeminiTimeoutError("Gemini request timed out") from exc
        except aiohttp.ClientError as exc:
            raise GeminiProviderError("Gemini request failed") from exc

        if not isinstance(payload, dict):
            raise GeminiInvalidResponseError("Gemini returned an invalid response")
        return payload


def parse_gemini_move(board: chess.Board, response_data: dict[str, Any]) -> chess.Move:
    text = _extract_response_text(response_data)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise GeminiInvalidResponseError("Gemini returned malformed move data") from exc

    move_text = data.get("move") if isinstance(data, dict) else None
    if not isinstance(move_text, str):
        raise GeminiInvalidResponseError("Gemini response did not contain a move")

    try:
        move = chess.Move.from_uci(move_text.strip().lower())
    except ValueError as exc:
        raise GeminiInvalidResponseError("Gemini returned an invalid UCI move") from exc

    if move not in board.legal_moves:
        raise GeminiInvalidResponseError("Gemini returned an illegal move")
    return move


def _extract_response_text(response_data: dict[str, Any]) -> str:
    try:
        parts = response_data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise GeminiInvalidResponseError("Gemini response contained no candidate") from exc

    texts = [
        part.get("text", "")
        for part in parts
        if isinstance(part, dict) and not part.get("thought") and isinstance(part.get("text"), str)
    ]
    text = "".join(texts).strip()
    if not text:
        raise GeminiInvalidResponseError("Gemini response contained no move data")
    return text


def _resolve_timeout(timeout_seconds: float | None) -> float:
    raw_value: float | str = timeout_seconds if timeout_seconds is not None else os.environ.get(
        "GEMINI_TIMEOUT_SECONDS",
        DEFAULT_GEMINI_TIMEOUT_SECONDS,
    )
    try:
        timeout = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise GeminiNotConfiguredError("Invalid Gemini timeout") from exc
    return max(5.0, min(120.0, timeout))

