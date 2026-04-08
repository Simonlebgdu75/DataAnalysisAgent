# `pe_qa` Vercel Demo Frontend

This repo now contains a standalone Next.js 16 App Router frontend for the `pe_qa` LangGraph demo.

## What it does

- protects `/app` and `/api/pe-qa/*` behind a shared app gate
- keeps all LangGraph credentials server-side
- exposes only same-origin BFF routes to the browser
- handles first-turn LangGraph interrupts through `/api/pe-qa/resume`
- renders a simple v1 UI with conversation + shortlist

The existing architecture notes for the backend graph are still available in [`docs/company-research/README.md`](docs/company-research/README.md).

## Routes

- `POST /api/auth/login`
- `POST /api/auth/logout`
- `GET /api/auth/session`
- `POST /api/pe-qa/message`
- `POST /api/pe-qa/resume`
- `GET /api/pe-qa/state?threadId=...`

## Required environment variables

Copy `.env.example` to `.env.local` for local work and set the same secrets in Vercel Project Settings for preview/production.

### App gate

- `APP_GATE_PASSWORD_HASH`
- `APP_GATE_SESSION_SECRET`

Generate the password hash with:

```bash
npm run hash:password -- "your-shared-password"
```

The output format is:

```text
scrypt$N$r$p$salt$hash
```

In `.env.local`, escape each `$` as `\$` because Next.js expands `$...` in env files.
In Vercel Project Settings, keep the raw value without backslashes.

### LangGraph backend

- `LANGGRAPH_BASE_URL`
- `LANGGRAPH_API_KEY`
- `LANGGRAPH_AUTH_HEADER` default: `x-api-key`
- `LANGGRAPH_AUTH_SCHEME` default: `Bearer`
- `LANGGRAPH_ASSISTANT_ID` default: `pe_qa`
- `LANGGRAPH_TIMEOUT_MS` default: `55000`

### Optional WAF rate-limit IDs

These are only used if you configure matching rate-limit rules in Vercel Firewall:

- `RATE_LIMIT_LOGIN_ID`
- `RATE_LIMIT_MESSAGE_ID`
- `RATE_LIMIT_RESUME_ID`

## Local development

```bash
npm install
npm run dev
```

Open `http://localhost:3000/login`.

## Vercel checklist

1. Create a dedicated Vercel project for this repo.
2. Add all sensitive env vars in Project Settings.
3. Enable Vercel Authentication for previews and deployment URLs.
4. Configure Vercel Firewall rate limits if you want the optional SDK checks to activate.
5. Point a private demo subdomain at the production deployment.

## Implementation notes

- `proxy.ts` protects `/app/:path*` and `/api/pe-qa/:path*`.
- Every BFF handler also re-checks auth server-side.
- POST routes enforce same-origin requests using the `Origin` header.
- The BFF normalizes LangGraph state and strips private fields like `_cg_messages` and `_agent_context`.
- No streaming, SSE, or WebSocket is used in v1.
