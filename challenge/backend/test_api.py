import asyncio
import json

import chess
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import api


def player(**changes):
    values = {"provider": "openai_compatible", "model": "test-model", "api_key": "top-secret"}
    values.update(changes)
    return api.PlayerConfig(**values)


def test_history_replays_legal_moves_and_rejects_invalid_history():
    board = api._board_from_moves(["e2e4", "e7e5"])
    assert board.fen() == chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2").fen()
    with pytest.raises(HTTPException, match="Invalid move history"):
        api._board_from_moves(["e2e5"])


def test_provider_payloads_keep_key_out_of_body_and_support_reasoning():
    for provider in ("openai_compatible", "gemini", "anthropic", "anthropic_compatible"):
        url, headers, body = api._request_details(player(provider=provider, reasoning="high", max_tokens=2048), chess.Board(), False)
        assert url.startswith("https://")
        assert "top-secret" in str(headers)
        assert "top-secret" not in json.dumps(body)
        if provider == "gemini":
            assert "test-model" in url
        else:
            assert body["model"] == "test-model"
        assert "e2e4" in json.dumps(body)
        if provider == "openai_compatible":
            assert body["reasoning_effort"] == "high"
        elif provider == "gemini":
            assert body["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 1536
        elif provider == "anthropic":
            assert body["thinking"] == {"type": "adaptive"}
            assert body["output_config"]["effort"] == "high"
        else:
            assert body["thinking"] == {"type": "enabled", "budget_tokens": 1024}


def test_custom_provider_url_must_use_https_and_no_credentials():
    assert api._allowed_base_url("https://example.com/v1") == "https://example.com/v1"
    for bad in (
        "http://example.com/v1", "https://name:pass@example.com/v1",
        "https://example.com:8443/v1", "https://example.com:invalid/v1",
        "https://127.0.0.1/v1", "https://[::1]/v1",
        "https://example.com/v1\nextra", "https://example.com/a b",
    ):
        with pytest.raises(HTTPException):
            api._allowed_base_url(bad)


def test_api_key_must_be_safe_for_http_headers_and_validation_never_echoes_it():
    for bad in ("leading space", "line\nbreak", "nonascii-\u00e9", "tab\tinside"):
        with pytest.raises(ValueError):
            player(api_key=bad)
        with TestClient(api.app) as client:
            response = client.post("/api/move", json={"player": {
                "provider": "openai_compatible", "model": "test-model", "api_key": bad,
            }})
        assert response.status_code == 422
        assert bad not in response.text
    assert player(api_key="key.with/+symbols_123").api_key == "key.with/+symbols_123"


def test_custom_compatible_provider_uses_common_token_parameter():
    _, _, custom = api._request_details(player(base_url="https://example.com/v1"), chess.Board(), False)
    _, _, openai = api._request_details(player(), chess.Board(), False)
    _, _, openrouter = api._request_details(player(base_url="https://openrouter.ai/api/v1"), chess.Board(), False)
    assert custom["max_tokens"] == 16000
    assert "max_completion_tokens" not in custom
    assert openai["max_completion_tokens"] == 16000
    assert openrouter["max_completion_tokens"] == 16000
    _, _, explicit = api._request_details(player(base_url="https://example.com/v1", token_parameter="max_completion_tokens"), chess.Board(), False)
    assert explicit["max_completion_tokens"] == 16000
    assert "max_tokens" not in explicit


def test_custom_anthropic_compatible_endpoint_uses_messages_protocol():
    url, headers, body = api._request_details(
        player(provider="anthropic_compatible", base_url="https://example.com/anthropic/v1"),
        chess.Board(), False,
    )
    assert url == "https://example.com/anthropic/v1/messages"
    assert headers["x-api-key"] == "top-secret"
    assert body["max_tokens"] == 16000


def test_both_colors_receive_same_oriented_board_and_legal_choices():
    white = api._prompt(chess.Board(), False)
    assert "Side to move: White" in white
    assert "8  r n b q k b n r" in white
    assert "1  R N B Q K B N R" in white
    assert "e2e4 (e4)" in white
    black = api._prompt(api._board_from_moves(["e2e4"]), False)
    assert "Side to move: Black" in black
    assert "8  r n b q k b n r" in black
    assert "1  R N B Q K B N R" in black
    assert "4  . . . . P . . ." in black
    assert "e7e5 (e5)" in black
    assert "1. e4" in black


def test_every_provider_receives_identical_chess_position_prompt():
    board = api._board_from_moves(["e2e4", "c7c5", "g1f3"])
    prompts = []
    for provider in ("openai_compatible", "gemini", "anthropic", "anthropic_compatible"):
        _, _, body = api._request_details(player(provider=provider), board, False)
        if provider == "gemini":
            prompts.append(body["contents"][0]["parts"][0]["text"])
        else:
            prompts.append(body["messages"][0]["content"])
    assert len(set(prompts)) == 1
    assert board.fen() in prompts[0]
    assert "Side to move: Black" in prompts[0]
    assert "d7d6 (d6)" in prompts[0]


def test_prompt_includes_promotion_choices_from_authoritative_board():
    board = chess.Board("8/P7/8/8/8/8/8/k1K5 w - - 0 1")
    prompt = api._prompt(board, False)
    assert "a7a8q (a8=Q#)" in prompt
    assert "a7a8n (a8=N)" in prompt


def test_prompt_includes_castling_and_en_passant_when_legal():
    castling = api._prompt(chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"), False)
    assert "e1g1 (O-O)" in castling
    assert "e1c1 (O-O-O)" in castling
    en_passant = api._prompt(api._board_from_moves(["e2e4", "a7a6", "e4e5", "d7d5"]), False)
    assert "e5d6 (exd6)" in en_passant


def test_public_resolver_rejects_private_addresses(monkeypatch):
    async def private_info(*args, **kwargs):
        return [(2, 1, 6, "", ("127.0.0.1", 443))]

    class Loop:
        getaddrinfo = private_info

    monkeypatch.setattr(api.asyncio, "get_running_loop", lambda: Loop())
    with pytest.raises(api.ProviderFailure, match="not a public address"):
        asyncio.run(api.PublicResolver().resolve("localhost", 443))


def test_move_parser_rejects_illegal_response():
    board = chess.Board()
    move, explanation = api._parse_move(board, '{"move":"e2e4","explanation":"Center"}')
    assert move.uci() == "e2e4" and explanation == "Center"
    with pytest.raises(api.ProviderFailure):
        api._parse_move(board, '{"move":"e2e5"}')


def test_provider_answers_ignore_private_thought_blocks_and_normalize_usage():
    gemini = {"candidates": [{"content": {"parts": [
        {"thought": True, "text": "hidden"},
        {"text": '{"move":"e2e4"}'},
    ]}}], "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 7}}
    answer, usage, _ = api._extract_answer("gemini", gemini)
    assert answer == '{"move":"e2e4"}'
    assert "hidden" not in answer
    assert usage == {"input_tokens": 12, "output_tokens": 7}
    anthropic = {"content": [{"type": "text", "text": '{"move":"e2e4"}'}],
                 "usage": {"input_tokens": 8, "output_tokens": 3}}
    assert api._extract_answer("anthropic", anthropic)[1] == {"input_tokens": 8, "output_tokens": 3}


def test_malformed_provider_usage_returns_controlled_error():
    with pytest.raises(api.ProviderFailure, match="no usable text"):
        api._extract_answer("openai_compatible", {"choices": [{"message": {"content": '{"move":"e2e4"}'}}], "usage": [1]})


@pytest.mark.asyncio
async def test_move_endpoint_returns_verified_position_without_key(monkeypatch):
    async def fake_ask(_session, _player, _board, _retry):
        return '{"move":"e2e4","explanation":"Controls the center"}', {"input_tokens": 10}, None

    monkeypatch.setattr(api, "_ask_provider", fake_ask)
    result = await api.move(api.MoveRequest(player=player()))
    assert result["move"] == "e2e4"
    assert result["san"] == "e4"
    assert result["explanation"] == "Controls the center"
    assert result["fen_after"] == api._board_from_moves(["e2e4"]).fen()
    assert "top-secret" not in json.dumps(result)


@pytest.mark.asyncio
async def test_move_retries_one_illegal_provider_answer(monkeypatch):
    attempts = []

    async def fake_ask(_session, _player, _board, retry):
        attempts.append(retry)
        return ('{"move":"e2e5"}' if not retry else '{"move":"e2e4"}'), {}, None

    monkeypatch.setattr(api, "_ask_provider", fake_ask)
    result = await api.move(api.MoveRequest(player=player()))
    assert result["move"] == "e2e4"
    assert attempts == [False, True]


@pytest.mark.asyncio
async def test_busy_server_rejects_without_waiting_for_provider(monkeypatch):
    monkeypatch.setattr(api, "REQUEST_LIMIT", asyncio.Semaphore(0))
    with pytest.raises(HTTPException, match="server is busy"):
        await api.move(api.MoveRequest(player=player()))


@pytest.mark.asyncio
async def test_move_reports_claimable_repetition_draw(monkeypatch):
    history = (
        "e2e4 e7e5 g1f3 b8c6 f3g5 g8e7 g5h7 h8g8 "
        "h7f8 g8h8 f8h7 h8g8 h7f8 g8h8 f8h7"
    ).split()

    async def fake_ask(_session, _player, _board, _retry):
        return '{"move":"h8g8","explanation":"Repeat"}', {}, None

    monkeypatch.setattr(api, "_ask_provider", fake_ask)
    result = await api.move(api.MoveRequest(player=player(), moves=history))
    assert result["result"] == "1/2-1/2"
    assert result["termination"] == "THREEFOLD_REPETITION"


def test_invalid_request_does_not_echo_api_key():
    with TestClient(api.app) as client:
        response = client.post("/move", json={"player": {"provider": "invalid", "model": "x", "api_key": "top-secret"}})
    assert response.status_code == 422
    assert "top-secret" not in response.text


def test_same_origin_api_routes_are_available():
    with TestClient(api.app) as client:
        assert client.get("/api/health").json()["ok"] is True
        response = client.post("/api/move", json={})
    assert response.status_code == 422
    assert response.headers["cache-control"] == "no-store"


def test_oversized_request_is_rejected_before_validation():
    with TestClient(api.app) as client:
        response = client.post("/api/move", content=b"x" * (api.MAX_REQUEST_BYTES + 1))
    assert response.status_code == 413


def test_embedded_private_ipv6_addresses_are_rejected():
    for bad in (
        "https://[::ffff:127.0.0.1]/v1", "https://[::ffff:10.0.0.1]/v1",
        "https://[2002:7f00:1::1]/v1", "https://[64:ff9b::a00:1]/v1",
    ):
        with pytest.raises(HTTPException):
            api._allowed_base_url(bad)
    assert api._is_public_ip(api.ipaddress.ip_address("2606:4700:4700::1111"))


def test_operator_allowlist_limits_provider_hosts(monkeypatch):
    monkeypatch.setattr(api, "ALLOWED_PROVIDER_HOSTS", ["api.openai.com", "*.example.com"])
    assert api._allowed_base_url(None) == "https://api.openai.com/v1"
    assert api._allowed_base_url("https://llm.example.com/v1") == "https://llm.example.com/v1"
    for bad in ("https://evil.com/v1", "https://example.com.evil.com/v1"):
        with pytest.raises(HTTPException, match="not allowed"):
            api._allowed_base_url(bad)
    with pytest.raises(HTTPException, match="not allowed"):
        api._request_details(player(provider="gemini"), chess.Board(), False)


def test_provider_error_statuses_map_to_safe_actionable_messages():
    assert api._status_failure(401).status == 401
    assert api._status_failure(429).status == 429
    assert "model" in api._status_failure(404).message
    assert api._status_failure(503).status == 503
    assert "redirect" in api._status_failure(307).message


def test_chunked_oversized_request_is_rejected_without_content_length():
    def chunks():
        for _ in range(5):
            yield b"x" * 4000

    with TestClient(api.app) as client:
        response = client.post("/api/move", content=chunks())
    assert response.status_code == 413


def test_valid_body_is_replayed_to_the_endpoint(monkeypatch):
    async def fake_ask(_session, _player, _board, _retry):
        return '{"move":"d2d4","explanation":"Center"}', {}, None

    monkeypatch.setattr(api, "_ask_provider", fake_ask)
    with TestClient(api.app) as client:
        response = client.post("/api/move", json={"player": player().model_dump(), "moves": ["e2e4", "e7e5"]})
    assert response.status_code == 200
    data = response.json()
    assert data["move"] == "d2d4" and data["attempts"] == 1
    assert data["fen_before"] == api._board_from_moves(["e2e4", "e7e5"]).fen()
    assert "top-secret" not in response.text


def test_security_headers_and_cache_policy():
    with TestClient(api.app) as client:
        response = client.get("/api/health")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["cache-control"] == "no-store"


def test_rate_limiter_blocks_after_limit_and_recovers():
    limiter = api.RateLimiter(2, window=60)
    assert limiter.retry_after("a", now=0) is None
    assert limiter.retry_after("a", now=1) is None
    assert limiter.retry_after("a", now=2) == 58
    assert limiter.retry_after("b", now=2) is None
    assert limiter.retry_after("a", now=61) is None


def test_move_endpoint_is_rate_limited(monkeypatch):
    monkeypatch.setattr(api, "MOVE_RATE_LIMIT", api.RateLimiter(1))
    with TestClient(api.app) as client:
        client.post("/api/move", json={})
        response = client.post("/api/move", json={})
    assert response.status_code == 429
    assert "retry-after" in response.headers


def test_disconnect_cancels_provider_work():
    cancelled = asyncio.Event()

    async def slow():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def gone():
        return True

    async def run():
        with pytest.raises(api.ProviderFailure):
            await api._cancel_on_disconnect(slow(), gone, interval=0.01)
        await asyncio.sleep(0)
        assert cancelled.is_set()

    asyncio.run(run())


def test_game_over_history_is_rejected():
    mate = ["f2f3", "e7e5", "g2g4", "d8h4"]
    with pytest.raises(HTTPException, match="Game is over"):
        api._board_from_moves(mate)


@pytest.mark.asyncio
async def test_checkmating_move_reports_result(monkeypatch):
    async def fake_ask(_session, _player, _board, _retry):
        return '{"move":"d8h4","explanation":"Mate"}', {}, None

    monkeypatch.setattr(api, "_ask_provider", fake_ask)
    result = await api.move(api.MoveRequest(player=player(), moves=["f2f3", "e7e5", "g2g4"]))
    assert result["result"] == "0-1" and result["termination"] == "CHECKMATE"


@pytest.mark.asyncio
async def test_two_invalid_answers_fail_without_applying_a_move(monkeypatch):
    async def fake_ask(_session, _player, _board, _retry):
        return "I like e4", {}, None

    monkeypatch.setattr(api, "_ask_provider", fake_ask)
    with pytest.raises(HTTPException, match="valid move"):
        await api.move(api.MoveRequest(player=player()))


def test_explanation_strips_control_characters():
    _, explanation = api._parse_move(chess.Board(), json.dumps({"move": "e2e4", "explanation": "a\u0007b\u001bc"}))
    assert explanation == "abc"


def test_claude_reasoning_uses_the_thinking_shape_each_model_accepts():
    def body(model, provider="anthropic", **changes):
        return api._request_details(player(provider=provider, model=model, reasoning="medium", **changes), chess.Board(), False)[2]

    for modern in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-4-8", "claude-sonnet-4-6", "claude-fable-5-1"):
        assert body(modern)["thinking"] == {"type": "adaptive"}, modern
        assert body(modern)["output_config"] == {"effort": "medium"}
    for legacy in ("claude-haiku-4-5", "claude-sonnet-4-5", "claude-sonnet-4-20250514", "claude-opus-4-1"):
        assert body(legacy)["thinking"] == {"type": "enabled", "budget_tokens": 6144}, legacy
        assert "output_config" not in body(legacy)
    third_party = body("kimi-k2", provider="anthropic_compatible", base_url="https://example.com/v1")
    assert third_party["thinking"]["type"] == "enabled"
    assert body("claude-opus-5-5", max_tokens=2048)["thinking"] == {"type": "adaptive"}
    assert "thinking" not in api._request_details(player(provider="anthropic", model="claude-opus-5-5"), chess.Board(), False)[2]
    with pytest.raises(HTTPException, match="at least 2048"):
        body("claude-haiku-4-5", max_tokens=1500)


def test_gemini_reasoning_uses_budget_or_level_by_generation():
    def config(model):
        return api._request_details(player(provider="gemini", model=model, reasoning="high"), chess.Board(), False)[2]["generationConfig"]

    assert config("gemini-2.5-flash")["thinkingConfig"] == {"thinkingBudget": 15488}
    assert config("gemini-3-pro-preview")["thinkingConfig"] == {"thinkingLevel": "HIGH"}


def test_lenient_parser_accepts_prose_fences_think_blocks_and_san():
    board = chess.Board()
    cases = [
        'Here is my move: {"move": "e2e4", "explanation": "Center"}',
        '```json\n{"move":"e2e4","explanation":"Center"}\n```',
        '<think>maybe {"move":"a2a3"}</think>{"move":"e2e4","explanation":"Center"}',
        '{"move":"d2d4","explanation":"draft"} final answer: {"move":"e2e4","explanation":"Center"}',
        '{"move":"e4","explanation":"Center"}',
    ]
    for text in cases:
        move, explanation = api._parse_move(board, text)
        assert move.uci() == "e2e4", text
        assert explanation == "Center"
    for bad in ('{"move":"e5"}', '{"move":"Ke2"}', "no json here", '{"explanation":"x"}'):
        with pytest.raises(api.ProviderFailure):
            api._parse_move(board, bad)


def test_stop_reasons_are_reported_per_provider():
    assert api._extract_answer("anthropic", {"content": [], "stop_reason": "max_tokens"})[2] == "length"
    assert api._extract_answer("anthropic", {"content": [], "stop_reason": "refusal"})[2] == "refusal"
    openai = {"choices": [{"message": {"content": None}, "finish_reason": "length"}]}
    assert api._extract_answer("openai_compatible", openai)[2] == "length"
    gemini = {"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]}
    assert api._extract_answer("gemini", gemini)[2] == "length"
    assert api._extract_answer("gemini", {"promptFeedback": {"blockReason": "SAFETY"}})[2] == "refusal"
    done = {"choices": [{"message": {"content": '{"move":"e2e4"}'}, "finish_reason": "stop"}]}
    assert api._extract_answer("openai_compatible", done)[2] is None


@pytest.mark.asyncio
async def test_truncated_answer_fails_once_with_actionable_message(monkeypatch):
    calls = []

    async def fake_ask(_session, _player, _board, retry):
        calls.append(retry)
        return '{"move":', {}, "length"

    monkeypatch.setattr(api, "_ask_provider", fake_ask)
    with pytest.raises(HTTPException) as caught:
        await api.move(api.MoveRequest(player=player()))
    assert caught.value.status_code == 422
    assert "Max output tokens" in caught.value.detail
    assert calls == [False]


@pytest.mark.asyncio
async def test_truncated_but_complete_answer_is_still_accepted(monkeypatch):
    async def fake_ask(_session, _player, _board, _retry):
        return '{"move":"e2e4","explanation":"Center"} and then more text that got cut', {}, "length"

    monkeypatch.setattr(api, "_ask_provider", fake_ask)
    assert (await api.move(api.MoveRequest(player=player())))["move"] == "e2e4"


@pytest.mark.asyncio
async def test_refusal_is_reported_without_retry(monkeypatch):
    calls = []

    async def fake_ask(_session, _player, _board, retry):
        calls.append(retry)
        return "", {}, "refusal"

    monkeypatch.setattr(api, "_ask_provider", fake_ask)
    with pytest.raises(HTTPException, match="declined"):
        await api.move(api.MoveRequest(player=player()))
    assert calls == [False]
