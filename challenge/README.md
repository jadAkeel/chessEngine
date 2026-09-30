# Chess Model Challenge

An independent application for watching two API models play chess against each other. It has its own frontend, backend, dependencies and run commands, and does not use the parent project's engine or model files.

For the public release, see [DEPLOYMENT.md](DEPLOYMENT.md). Visitors use their own provider keys. A separate, clearly labelled scripted opening replay lets them explore the board without a key or provider requests; it never enters real match scores or exports.

## How a match works

1. Configure **Model A** and **Model B**: provider or compatible protocol, base URL (custom providers), model ID, API key, and optional reasoning effort, output-token limit and timeout.
2. **Test connection** asks the provider for one opening move and verifies it. Nothing is recorded; the provider may bill the request.
3. **Start match** plays automatically; **One move** plays a single half-move. The default format is two games with colors swapped (A is White in game 1, B is White in game 2). A single game is also available.
4. The browser asks the challenge API for one move at a time. The API rebuilds the position from the UCI history, sends the side to move the same prompt (board diagram, FEN, UCI and SAN history, every legal move) and accepts only a legal move. It reads the last JSON object containing `"move"` in the reply, ignoring surrounding prose, code fences and `<think>` blocks, and accepts UCI or unambiguous SAN; the same rules apply to both players. One invalid answer gets one corrected retry, shown as "retry" in the move list.
5. The browser applies the move only if the server's before/after FEN matches its own board. Results come from the server.
6. Score, per-game results, move list, board review (click a move or use ← →, Esc returns to live), PGN export (with rationales as comments) and a key-free JSON report are available throughout.

Settings lock after the first move so every move of a match uses the same configuration. While paused, an API key can still be replaced (for example after a quota error); reset the match to change anything else.

Explanations are **public move rationales written by each model**, not private chain-of-thought.

### Reasoning and token budget

`Max output tokens` (default 16000, up to 64000) covers reasoning tokens too, so keep it generous for reasoning models; some providers cap it lower and answer HTTP 400, in which case lower it. The default timeout is 120 s (up to 600). With `Provider default` nothing extra is sent, and models differ: some reason by default, others do not. Other levels map to:

| Provider | Sent for low / medium / high |
| --- | --- |
| OpenAI-compatible | `reasoning_effort` |
| Anthropic, Claude 4.6+ (and unrecognized official IDs) | `thinking: {type: "adaptive"}` + `output_config.effort` |
| Anthropic, Claude 4.5 and older, Haiku | `thinking.budget_tokens` 2048 / 6144 / 12288, capped at max tokens − 1024 |
| Anthropic-compatible, non-Claude model IDs | `thinking.budget_tokens` as above (third-party support varies) |
| Gemini 3+ | `thinkingConfig.thinkingLevel` |
| Gemini 2.x | `thinkingConfig.thinkingBudget` 1024 / 8192 / 24576, capped below max tokens |

A reply cut off by the token limit, or declined by the provider's safety system, fails the turn immediately with that reason instead of a generic error.

The UI warns when the two models use different reasoning, token or timeout settings, and the report records `settings_match` and the differences.

### Failures and recovery

A failed provider call stops the match without assigning a loss. Every failed request is recorded with its model, ply and reason; the score card and the JSON report (`reliability`, per-game `failures`) show failed requests and moves that needed a retry for each model, so recovery by retrying does not hide an unreliable model. The error panel names the model and suggests a fix (key rejected, rate limit, timeout/busy, bad request, unreachable server, board mismatch); **Retry move** resends the same position. Pausing aborts the in-flight request, and the server cancels the provider call when the browser disconnects, though a provider may still bill a request it already received.

## Providers

| Option | Endpoint the app calls | Auth header |
| --- | --- | --- |
| Custom · OpenAI-compatible (default) | `POST {base URL}/chat/completions` | `Authorization: Bearer` |
| Custom · Anthropic-compatible | `POST {base URL}/messages` | `x-api-key` |
| OpenAI | `https://api.openai.com/v1/chat/completions` | `Authorization: Bearer` |
| OpenRouter | `https://openrouter.ai/api/v1/chat/completions` | `Authorization: Bearer` |
| Google Gemini | `…/v1beta/models/{model}:generateContent` | `x-goog-api-key` |
| Anthropic | `https://api.anthropic.com/v1/messages` | `x-api-key` |

The setup card shows the exact endpoint for the current settings. The base-URL field suggests examples (Kimi, Z.ai/GLM, DeepSeek, Mistral, Groq, Together, xAI); confirm the URL and model ID in your provider's documentation for your account and region. For Anthropic-compatible providers that document an SDK `ANTHROPIC_BASE_URL`, append `/v1`, because this app appends only `/messages`.

For OpenAI-compatible endpoints, **Output token field** chooses `max_tokens` or `max_completion_tokens`. Automatic sends `max_tokens` to custom URLs and `max_completion_tokens` to the OpenAI and OpenRouter presets.

### Example: Gemini versus Atria

For Gemini, select **Google Gemini**, enter a model available to your key (for example `gemini-3.6-flash`), and paste the Google API key into that player's key field. For Atria, select **Custom · OpenAI-compatible**, set the base URL to `https://api.atria-asi.ai/v1`, model ID to `Atria-Dawn-Preview`, and enter an Atria API key. Use the same reasoning level, output-token limit and timeout for both players, then run **Test connection** for each before starting a paired match. Model availability and provider support can change; use the IDs and limits offered by your account.

OpenCode's locally stored provider credentials are not imported by this web application. Each player enters a key in the browser tab for the current session.

## Security model

- API keys live only in page memory. They are not written to local/session storage, cookies, URLs, logs or exports, and API errors never echo request bodies. Reloading the page clears them and the in-progress match.
- The key is sent to this server with each move request and forwarded to the endpoint the user configured. Run the service only over HTTPS.
- Custom URLs must be `https://` on port 443 without credentials, query or fragment. The server resolves every hostname and connects only to globally routable addresses (including checks on IPv4-mapped, 6to4, Teredo and NAT64 forms), ignores proxy environment variables and does not follow redirects. Operators can restrict outbound hosts with `CHALLENGE_ALLOWED_PROVIDER_HOSTS`.
- Request bodies are capped at 16 KiB, including chunked uploads. Move requests are rate limited per client address and limited to a small number of concurrent provider calls. Provider responses are capped at 1 MB.
- Responses carry a strict Content-Security-Policy, `X-Frame-Options: DENY`, `nosniff`, `Referrer-Policy: no-referrer` and `Cache-Control: no-store` on API routes. The UI loads no third-party scripts or fonts.
- In the single-service deployment the UI and API share an origin, so CORS is unused; cross-origin browser calls are refused unless listed in `CHALLENGE_CORS_ORIGINS`.

## Run locally

Use Python 3.11+ and Node.js 20.19+ (the image uses Node 22). In two terminals:

```powershell
cd challenge/backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn api:app --reload --host 127.0.0.1 --port 8001
```

```powershell
cd challenge/frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The dev server proxies `/api` to `http://127.0.0.1:8001` (override with `CHALLENGE_API_PROXY`), so the browser uses the same origin as in production.

To serve the built UI from the API, as the image does: run `npm run build`, set `CHALLENGE_STATIC_DIR` to the absolute path of `challenge/frontend/dist`, and start uvicorn.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` | `8000` | Listening port in the image. |
| `CHALLENGE_STATIC_DIR` | `/app/static` in the image | Serve the built UI from this directory. |
| `FORWARDED_ALLOW_IPS` | `127.0.0.1` | Proxies whose `X-Forwarded-For` is trusted. Set to your load balancer's address, or `*` only when the platform proxy is the sole route to the container. Otherwise every visitor shares the proxy's rate-limit bucket. |
| `CHALLENGE_RATE_LIMIT_PER_MINUTE` | `60` | Move requests per client address per minute (`0` disables). In-process; each instance counts separately. |
| `CHALLENGE_MAX_CONCURRENT_MOVES` | `4` | Concurrent provider calls per instance; extra requests get HTTP 503 and can be retried. |
| `CHALLENGE_ALLOWED_PROVIDER_HOSTS` | unset (any public host) | Comma-separated host allowlist, `*.example.com` wildcards allowed. Applies to presets too, so include e.g. `api.openai.com` if you use it. |
| `CHALLENGE_HSTS` | unset | Set to `1` to send `Strict-Transport-Security` when served only over HTTPS. |
| `CHALLENGE_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Exact browser origins allowed to call the API cross-origin. Only needed for split hosting. |
| `VITE_CHALLENGE_API_BASE_URL` | unset (`/api`) | Build-time API location for a separately hosted UI. Leave unset for the image. |

## Deploy

Build the [Dockerfile](Dockerfile) from `challenge/` and run it as one HTTPS web service. It builds the UI, installs the pinned backend, runs as a non-root user, and serves UI and API on `PORT` with a `/api/health` container health check.

```powershell
cd challenge
docker build -t chess-model-challenge .
docker run --rm -p 8000:8000 chess-model-challenge
```

Then open `http://localhost:8000` and check `http://localhost:8000/api/health`.

On any container platform, terminate TLS at the platform, route to `PORT`, use `/api/health` as the health check, and set `FORWARDED_ALLOW_IPS` (see above). For Render, [render.yaml](render.yaml) describes the service: create a Blueprint in the dashboard with the Blueprint path `challenge/render.yaml`; it selects branch `codex/challenger`, root directory `challenge/`, the Dockerfile, health check, HSTS and proxy trust. It uses the paid `0.5c-512mb` plan (legacy name `starter`) and `autoDeploy: false`, so trigger deploys manually. [render.preview.yaml](render.preview.yaml) offers a Free public preview with sleep and usage limits; see [DEPLOYMENT.md](DEPLOYMENT.md) for the release process.

Before a public launch, also:

- Run one instance or accept per-instance rate limits; put platform rate limiting or a WAF in front for public traffic.
- Consider `CHALLENGE_ALLOWED_PROVIDER_HOSTS`. Without it the service relays requests with user-supplied keys to any public HTTPS host.
- Monitor HTTP 429/503/504 rates and latency. Access logs contain paths and status codes only.
- The service has no accounts, persistence or leaderboard. Review the GPL-3.0 license of the `chess` package before distributing the image.

## Benchmark rules and limits

- Both players get the same prompt template (`board-fen-legal/v1`), board orientation (files a–h left to right, ranks 8–1 top to bottom, uppercase White), side to move, FEN, UCI and SAN history and all legal moves.
- The server decides legality, termination and results, and claims threefold-repetition and 50-move draws automatically. Games stop at 300 half-moves and are then scored as draws.
- A pair of games reduces first-move bias but is one sample, not a ranking. Model nondeterminism, token budgets, timeouts and provider reasoning support affect comparisons. Keep settings fixed and run multiple pairs before publishing claims.
- The JSON report is assembled in the browser. It is useful for review, not tamper-proof evidence.

## Verification

```powershell
cd challenge/backend
pip install -r test_requirements.txt
python -m pytest -q
```

```powershell
cd challenge/frontend
npm test
npm run build
npm audit
```

Browser journeys run against the real API with scripted providers ([e2e/mock_api.py](e2e/mock_api.py), never deployed). Only the outbound provider call is replaced. After `npm run build`:

```powershell
python challenge/e2e/mock_api.py
```

Then, with `playwright-core` installed outside the project and a Chromium executable, run `challenge/e2e/journeys.mjs` with `BASE_URL=http://127.0.0.1:8011` and `CHROME_PATH` set. The journeys cover setup validation, connection tests, the SSRF guard, a paired match with color swap and score attribution, exports without keys, pause/resume/reset during a request, provider failure and retry, key rejection, FEN mismatch, network loss, mobile layout, and keyboard access.
