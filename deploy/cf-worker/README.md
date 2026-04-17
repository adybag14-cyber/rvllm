# Cloudflare Worker: solidsf.com/rvllm/docs/*

Serves `docs/` at `solidsf.com/rvllm/*` via Cloudflare Workers + Static Assets.
Auto-deploy on push to `main` via GitHub Action (see `.github/workflows/`).

## One-time setup

```bash
# From the repo root:
cd deploy/cf-worker
npm install -g wrangler
wrangler login          # browser login to your Cloudflare account
```

Prereq: the `solidsf.com` zone must already be on your Cloudflare account
(DNS NS records point at Cloudflare). If not yet — use `workers.dev` route
for testing and switch to the custom route after the zone is added.

## Deploy (manual)

```bash
bash build.sh    # stages docs/ into site/docs/
wrangler deploy
```

That's it. The Worker is live at `https://solidsf.com/rvllm/*` in <10 s.

## Test without the custom domain

Comment the `[[routes]]` block in `wrangler.toml` and deploy — the Worker
is reachable at `https://rvllm-docs.<account>.workers.dev/rvllm/docs/bench.html`.

## URL layout

Matches the repo one-to-one after stripping `/rvllm`:

| URL                                         | File served          |
| ------------------------------------------- | -------------------- |
| `solidsf.com/rvllm`                         | redirect to `/rvllm/docs/index.html` |
| `solidsf.com/rvllm/docs/index.html`         | `docs/index.html`    |
| `solidsf.com/rvllm/docs/bench.html`         | `docs/bench.html`    |
| `solidsf.com/rvllm/docs/paper/rvllm.pdf`    | `docs/paper/rvllm.pdf` |

## What got built

- `wrangler.toml` — Worker config, Static Assets binding, route pattern
- `src/worker.js` — request handler: strips `/rvllm` prefix, forwards to assets
- `build.sh` — rsyncs repo `docs/` into `./site/docs/` for deploy
- `site/` — build output, git-ignored

## Troubleshooting

- **`wrangler deploy` complains about `zone not found`**: the `solidsf.com`
  zone isn't on your Cloudflare account yet. Either add it (point NS
  records at CF) or remove the `[[routes]]` block and use the
  `workers.dev` subdomain for now.
- **404 at `solidsf.com/rvllm/`**: a Page Rule or existing Worker might be
  shadowing the route. `wrangler tail` shows whether the Worker fired.
- **Stale content after commit**: the Worker uses edge caching. HTML is
  `no-cache`, so changes are immediate. PDFs cache 1 h — bust with
  `wrangler deploy` (each deploy invalidates the assets binding).
