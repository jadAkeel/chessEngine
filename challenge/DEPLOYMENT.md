# Public launch on Render

The public release runs the frontend and API in one Docker web service. Visitors enter their own provider keys. No shared model key, account system, database, or parent chess-engine files are required.

## Published preview

- Public URL: [Chess Model Challenge](https://chess-model-challenge-preview.onrender.com)
- Service: [Render dashboard](https://dashboard.render.com/web/srv-dauhf7h42hec73f35640)
- Published: 2026-09-30, Free preview plan.
- Deployed application commit: `c6ee2a3395f3843e8d56f714e550315ee155b0df`.
- [Release workflow](https://github.com/jadAkeel/chessEngine/actions/runs/36726506057): frontend/backend checks, Linux image build and container smoke check passed.

The public URL passed browser checks for the desktop/mobile UI, sample isolation, reload, health, security headers and private-provider URL rejection. No model inference calls were made during the public smoke check. Local verification passed 41 backend tests, 12 frontend tests and 11 browser journeys with scripted providers.

The paid template is ready but has not been provisioned. Render reported `need_payment_info` when validating it against the account. Moving to an always-on paid plan requires the operator's cost approval and payment information.

## Release configuration

- Repository: `https://github.com/jadAkeel/chessEngine`
- Release branch: `codex/challenger`
- Root directory: `challenge`
- Runtime: Docker, Dockerfile `./Dockerfile`, build context `.`
- Health check: `/api/health`
- Region: Frankfurt
- UI and API share one HTTPS origin. Leave `VITE_CHALLENGE_API_BASE_URL` unset.

Use `challenge/render.yaml` for an always-on production instance on the paid `0.5c-512mb` plan (legacy name `starter`). The account must have a payment method before Render accepts this plan. Confirm the current charge on [Render pricing](https://render.com/pricing) before creating it.

Use `challenge/render.preview.yaml` for a public Free preview. [Render's Free limitations](https://render.com/docs/free) include sleep after 15 minutes of inactivity, about one minute to wake, shared monthly instance hours and restrictions on service-initiated traffic. This is a trial environment rather than an always-on production service.

## Publish

1. Commit and push the release branch. Run the `Chess challenge release checks` GitHub workflow; it verifies the frontend, backend, Linux Docker build and container health/UI.
2. In Render, create a **Blueprint** from the repository and select `codex/challenger`. Set the Blueprint path to the chosen YAML file. Review the service and compute charge, then create it.
3. Wait for the deploy to become **Live**, then open the service's `onrender.com` HTTPS URL.
4. Check `/api/health`, open the sample replay, and confirm the sample makes no provider requests. Test a real move using a visitor-owned key if the provider account has available quota.

The service has manual deployment enabled. After updates pass the workflow, use **Manual Deploy → Deploy latest commit** on Render. Adding a custom domain later does not require rebuilding the app.

The equivalent CLI creation for a Free preview is:

```powershell
render services create --name chess-model-challenge-preview --type web_service --runtime docker --repo https://github.com/jadAkeel/chessEngine --branch codex/challenger --root-directory challenge --plan free --region frankfurt --health-check-path /api/health --auto-deploy=false --env-var FORWARDED_ALLOW_IPS=* --env-var CHALLENGE_HSTS=1 --env-var CHALLENGE_CORS_ORIGINS= --env-var CHALLENGE_RATE_LIMIT_PER_MINUTE=60 --env-var CHALLENGE_MAX_CONCURRENT_MOVES=4 --output json
```

For the paid production service, use the production service name and `--plan 0.5c-512mb` after the operator approves the charge and adds payment information.

## What visitors can expect

The opening sample is scripted and never contributes to real match scores or reports. A real match uses two visitor-configured providers. Each API call, including connection tests and retries, can be billed by that provider. Keys are held in the browser tab and forwarded through the challenge API; the service does not persist them.

Matches run while the visitor keeps the tab open. Reloading loses the current match and clears keys. Visitors can export PGN and JSON, including failed-request counts. Provider timeouts, quota errors and unavailable models stop a match for retry and never assign a chess loss. A pair of color-swapped games is a sample of chess performance, not a general intelligence ranking.

## Operations and rollback

Start with one instance and four concurrent provider calls. Monitor Render request latency and HTTP 429/503/504 rates. Rate limits are per instance, so use a shared limiter before scaling to multiple instances. No provider keys should be added as Render environment variables.

To roll back, select a known successful deploy in Render and use **Rollback**. The service has no migrations or stored matches to restore. Existing visitors may need to reload the page after a frontend rollback.
