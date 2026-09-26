import json

import chess
import pytest
from fastapi import HTTPException

from app.api import main as api_main
from app.opponents.gemini import (
    DEFAULT_GEMINI_MODEL,
    GeminiInvalidResponseError,
    GeminiMove,
    GeminiNotConfiguredError,
    GeminiOpponent,
    parse_gemini_move,
)


def _response(move: str, *, include_thought: bool = False) -> dict:
    parts = []
    if include_thought:
        parts.append({"thought": True, "text": "private reasoning"})
    parts.append({"text": json.dumps({"move": move})})
    return {"candidates": [{"content": {"parts": parts}}]}


def test_gemini_defaults_to_requested_model_and_high_thinking(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_THINKING_LEVEL", raising=False)

    opponent = GeminiOpponent(api_key="test-key")

    assert opponent.model == DEFAULT_GEMINI_MODEL == "gemini-3.8-flash"
    assert opponent.thinking_level == "high"


def test_payload_constrains_response_to_legal_moves_without_api_key():
    board = chess.Board()
    opponent = GeminiOpponent(api_key="secret-test-key")

    payload = opponent.build_payload(board)
    generation_config = payload["generationConfig"]
    schema = generation_config["responseJsonSchema"]

    assert generation_config["responseMimeType"] == "application/json"
    assert generation_config["thinkingConfig"] == {
        "thinkingLevel": "HIGH",
        "includeThoughts": False,
    }
    assert schema["required"] == ["move"]
    assert set(schema["properties"]["move"]["enum"]) == {
        move.uci() for move in board.legal_moves
    }
    assert "secret-test-key" not in json.dumps(payload)


def test_parse_gemini_move_accepts_a_legal_uci_move_and_skips_thought_parts():
    board = chess.Board()

    move = parse_gemini_move(board, _response("e2e4", include_thought=True))

    assert move == chess.Move.from_uci("e2e4")


@pytest.mark.parametrize("response_data", [_response("e2e5"), _response("not-a-move"), {}])
def test_parse_gemini_move_rejects_invalid_provider_output(response_data):
    with pytest.raises(GeminiInvalidResponseError):
        parse_gemini_move(chess.Board(), response_data)


@pytest.mark.asyncio
async def test_choose_move_requires_backend_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(GeminiNotConfiguredError):
        await GeminiOpponent().choose_move(chess.Board())


@pytest.mark.asyncio
async def test_choose_move_retries_one_invalid_move(monkeypatch):
    opponent = GeminiOpponent(api_key="test-key")
    responses = iter([_response("e2e5"), _response("e2e4")])
    retry_flags = []

    async def fake_request(session, board, *, retry):
        retry_flags.append(retry)
        return next(responses)

    monkeypatch.setattr(opponent, "_request", fake_request)

    result = await opponent.choose_move(chess.Board())

    assert result.move == chess.Move.from_uci("e2e4")
    assert retry_flags == [False, True]


@pytest.mark.asyncio
async def test_provider_sends_api_key_in_header_not_url():
    captured = {}

    class FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def json(self, content_type=None):
            return _response("e2e4")

    class FakeSession:
        def post(self, url, *, headers, json):
            captured.update(url=url, headers=headers, payload=json)
            return FakeResponse()

    opponent = GeminiOpponent(api_key="secret-test-key")
    await opponent._request(FakeSession(), chess.Board(), retry=False)

    assert "secret-test-key" not in captured["url"]
    assert captured["headers"]["x-goog-api-key"] == "secret-test-key"
    assert "secret-test-key" not in json.dumps(captured["payload"])


@pytest.mark.asyncio
async def test_gemini_endpoint_returns_server_calculated_fen(monkeypatch):
    class FakeOpponent:
        async def choose_move(self, board):
            return GeminiMove(
                move=chess.Move.from_uci("e2e4"),
                model="gemini-3.8-flash",
                thinking_level="high",
            )

    monkeypatch.setattr(api_main, "GeminiOpponent", FakeOpponent)

    response = await api_main.gemini_move(api_main.GeminiMoveRequest(fen=chess.STARTING_FEN))

    expected = chess.Board()
    expected.push_uci("e2e4")
    assert response["move"] == "e2e4"
    assert response["san"] == "e4"
    assert response["source"] == "gemini"
    assert response["fen_after"] == expected.fen()


@pytest.mark.asyncio
async def test_gemini_endpoint_maps_missing_configuration_to_503(monkeypatch):
    class MissingOpponent:
        async def choose_move(self, board):
            raise GeminiNotConfiguredError("missing")

    monkeypatch.setattr(api_main, "GeminiOpponent", MissingOpponent)

    with pytest.raises(HTTPException) as exc_info:
        await api_main.gemini_move(api_main.GeminiMoveRequest(fen=chess.STARTING_FEN))

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Gemini API is not configured"


def test_health_reports_gemini_without_leaking_api_key(monkeypatch):
    secret = "super-secret-gemini-key"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_THINKING_LEVEL", raising=False)

    response = api_main.health()

    assert response["gemini"]["configured"] is True
    assert response["gemini"]["model"] == "gemini-3.8-flash"
    assert response["gemini"]["thinking_level"] == "high"
    assert secret not in json.dumps(response)
