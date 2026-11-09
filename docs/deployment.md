# Deployment

## Vercel frontend

The root [`vercel.json`](../vercel.json) keeps the repository root as the
Vercel project directory and configures these commands:

```text
installCommand: npm --prefix frontend ci
buildCommand: npm --prefix frontend run build
outputDirectory: frontend/dist
```

The ignored-build check uses `frontend/ignore-vercel-build.sh`: it skips
deployments when only files outside `frontend/` and the root `vercel.json`
change. Frontend changes and Vercel configuration changes continue to build.
If using the `frontend` Root Directory fallback, `frontend/vercel.json` invokes
the same script scoped to that directory.

It also rewrites client-side routes to `/index.html`. Configure
`VITE_API_URL` in the Vercel Preview and Production environments as appropriate;
Vite embeds this value at build time, so redeploy after changing it.

If the preview build does not work with the root project configuration, use
this dashboard fallback:

1. Set the Vercel project's **Root Directory** to `frontend`.
2. Keep **Include source files outside of the Root Directory** disabled.
3. Use `npm ci` for Install Command and `npm run build` for Build Command, with
   `dist` as the output directory.
4. Keep `frontend/vercel.json` in place for the SPA rewrite.

## FastAPI backend

Build and run the CPU-only image from the repository root for local testing:

```powershell
Copy-Item backend\.env.example backend\.env
docker build -t shortlist-backend backend
docker run --rm -p 8000:8000 --env-file backend\.env shortlist-backend
```

The image bakes `sentence-transformers/all-MiniLM-L6-v2` and runs the app
offline. Do not override `EMBED_MODEL` on the host: a different model must be
baked into a new image. The entrypoint performs any configured private
fine-tuned model download before setting the app process offline. It runs one
Uvicorn worker and does not trust Uvicorn proxy headers; the application
resolves the client address itself for rate limiting.

### Deploy on any Docker host

The backend can run on any Docker host with Docker Compose, sufficient memory
for CPU torch and the embedding model, and a reverse proxy that terminates TLS.
The following is an example only; replace the placeholders and adapt the proxy
configuration to your host. Keep the Compose project in a Git checkout on the
host, with `backend/.env` created there and excluded from source control.

```yaml
services:
  backend:
    build: ./backend
    env_file: ./backend/.env
    # Optional: mount a pre-provisioned fine-tuned model folder read-only.
    # Keep FINETUNED_REPO empty in backend/.env when using this mount.
    # Uncomment and set MODEL_PATH in the host environment or Compose .env:
    # volumes:
    #   - type: bind
    #     source: ${MODEL_PATH:?Set MODEL_PATH to the host model folder}
    #     target: /app/backend/models/finetuned
    #     read_only: true
    expose:
      - "8000"
    restart: unless-stopped

  proxy:
    image: ${PROXY_IMAGE:?Set PROXY_IMAGE to a reverse-proxy image}
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - type: bind
        source: ${PROXY_CONFIG:?Set PROXY_CONFIG to the host proxy config}
        target: ${PROXY_CONFIG_TARGET:?Set the proxy config container path}
        read_only: true
    depends_on:
      backend:
        condition: service_healthy
    restart: unless-stopped
```

Choose one fine-tuned model provisioning option:

1. **Pre-provisioned local folder:** put the complete, validated model folder
   (including `calibrator.pkl`) on the host, set `MODEL_PATH` to that folder,
   mount it read-only as above, and leave `FINETUNED_REPO` empty in
   `backend/.env`. The backend uses a valid mounted model without modifying it.
2. **Private Hub download:** omit the model bind mount and configure
   `FINETUNED_REPO`, optionally `FINETUNED_REVISION`, and `HF_TOKEN` in the
   host's `backend/.env`. The token should be read-only and kept out of source
   control. The configured folder is downloaded at container startup, so do
   not combine this option with a read-only model mount.

Set `TRUSTED_PROXY_COUNT` to the number of trusted proxy hops in the chain that
sets `X-Forwarded-For`; configure the proxy to overwrite or safely append that
header. The Docker image's `/health` check has a 180-second start period to
allow model loading. Compose deployments should wait for the backend health
status before considering a rollout complete. The example uses Compose's
`service_healthy` dependency for proxy startup; it does not restart or reload
the proxy on a backend-only deployment.

#### CI automatic deployment

The workflow deploy job is opt-in and runs after the backend tests and Docker
build pass, only for pushes to `master` that change `backend/**`. Configure
these GitHub repository secrets:

| Secret | Purpose |
| --- | --- |
| `DEPLOY_HOST` | Docker host address |
| `DEPLOY_USER` | SSH user with access to the Compose project and Docker |
| `DEPLOY_SSH_KEY` | Private SSH key for that user |
| `DEPLOY_PORT` | Optional SSH port; defaults to `22` |

Configure these GitHub repository variables:

| Variable | Purpose |
| --- | --- |
| `DEPLOY_ENABLED` | Set to `true` to enable deployment |
| `DEPLOY_PATH` | Absolute server path to the Compose project checkout |
| `COMPOSE_SERVICE` | Optional Compose service name; defaults to `backend` |
| `API_URL` | Optional public HTTPS API base URL for the post-deploy smoke test |

Do not set `COMPOSE_SERVICE` to `caddy` or another proxy service; the workflow
refuses to operate on the `caddy` service.

The SSH user needs Git access to the checkout and permission to run Docker.
Deploys use `git pull --ff-only origin master`, build and recreate only the
selected service, wait up to five minutes for its container health check, and
retain the preceding image as `:previous` for rollback. On failure, the
workflow prints the selected service's recent logs and attempts to restore the
previous image. If no previous backend image exists on an initial deployment,
it stops the failed backend service instead. It prunes dangling images only
and does not remove volumes or restart other services. The optional smoke test
checks `/health` and submits a synthetic PDF to `/analyze` through `API_URL`.

#### Hugging Face Space option

Hugging Face Spaces remain one supported backend hosting option. Follow the
[Hugging Face Space deployment instructions](hf-space-readme.md). Use the
private Hub download option for fine-tuned models there; the host bind mount
shown in the Compose example is specific to Docker hosts that provide a local
model folder.

### Private fine-tuned model updates

Training and calibration run locally, never on the server. The training output
is `backend\models\finetuned`; it contains the SentenceTransformer model and
`calibrator.pkl`. The complete folder can be mounted read-only on a Docker host
or uploaded to a private Hugging Face model repo for startup download. Model
updates do not require a Docker rebuild or a code change:

```powershell
Set-Location backend
python -m ml.train
python -m ml.calibrate
python -m app.fetch_model --check models\finetuned
Set-Location ..
hf upload YOUR_USER/resume-screener-model backend\models\finetuned .
```

Run the local checker before every upload. It validates the folder manifest,
Transformer files, and root `calibrator.pkl` without downloading or loading the
model. If an older training run left `models/calibrator.pkl`, copy it into
`models/finetuned/` (or a staging copy of that folder) before running `hf upload`.

Set these backend environment variables in the hosting provider's configuration
or in the host-only `backend/.env` file:

| Variable | Default | Purpose |
| --- | --- | --- |
| `ALLOWED_ORIGINS` | `https://pushcv.vercel.app,https://statcv.vercel.app,https://pullcv.vercel.app,https://taphire.vercel.app,https://dragcv.vercel.app,https://shortlists.vercel.app,http://localhost:5173,http://localhost:8080` | Comma-separated exact browser origins |
| `ALLOWED_ORIGIN_REGEX` | empty | Optional origin regex, e.g. `^https://pushcv(-[a-z0-9-]+)*\.vercel\.app$` |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Model baked into the image |
| `DEBUG_ENDPOINTS` | `0` | Enables optional score explanations |
| `MAX_PDF_BYTES` | `10485760` | Maximum bytes per PDF (10 MiB) |
| `MAX_PDF_PAGES` | `20` | Maximum pages per PDF |
| `MAX_REQUEST_BYTES` | `62914560` | Maximum request size (60 MiB) |
| `MAX_FILES` | `30` | Maximum PDFs per request |
| `MAX_CONCURRENT_ANALYSES` | `2` | Active requests per process |
| `QUEUE_TIMEOUT_SECONDS` | `30` | Wait time for an analysis slot |
| `RATE_LIMIT_PER_MINUTE` | `20` | Per-client analyze limit; `0` disables it |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `FINETUNED_REPO` | empty | Private Hub repo ID; empty makes no startup download (a complete local model folder is still used if present) |
| `FINETUNED_REVISION` | empty | Hub revision; empty uses latest. Pin an approved model to a commit hash |
| `HF_TOKEN` | empty | Read-only token with access to the private model repo; keep it out of source control |
| `TRUSTED_PROXY_COUNT` | `1` | Selects the trusted client address from the right of `X-Forwarded-For`; `0` ignores that header |

For production, set `ALLOWED_ORIGINS` to the actual frontend origin(s) that
should call the API.

To determine `TRUSTED_PROXY_COUNT` for your proxy chain:

1. Temporarily set `LOG_LEVEL=DEBUG`.
2. Send one `/analyze` request **without** an `X-Forwarded-For` header.
3. Read the `forwarded_for_entries` count in the request's debug log.
4. Set `TRUSTED_PROXY_COUNT` to that count, then return `LOG_LEVEL` to `INFO`.

The log includes only the entry count and a short hash of the resolved client,
never the raw IP. A wrong proxy count weakens per-client enforcement: a value
that is too high can prevent selecting the actual client address and may allow
the intended limit to be bypassed in a proxy chain; `0` or a value that is
too low can make visitors share a limit.

One Uvicorn process is intentional: every worker would load another embedding
model and copy of the 10k-skill lexicon, multiplying memory use. Host the
container on a Docker-capable service with enough RAM for CPU torch, MiniLM,
and request processing; terminate TLS with HTTPS, provide the environment
variables, and configure the platform health probe at `/health`. Allow the
image's 180-second Docker health-check start period for loading the model and
lexicon. Use a keep-alive or minimum-instance option if cold starts are
undesirable.

Set `VITE_API_URL` in the Vercel Production and Preview environments to the
public HTTPS API base URL, then redeploy because Vite embeds it at build time.
The active Vercel configuration is the root [`vercel.json`](../vercel.json)
when the Vercel Root Directory is the repository root. Keep
[`frontend/vercel.json`](../frontend/vercel.json) as the dashboard fallback
when the Root Directory is `frontend`; do not delete either file.

Check a live API with:

```powershell
python backend\tools\smoke_test.py --url https://api.example.com
```

See [local testing with ngrok](local-testing-with-ngrok.md) for temporary
preview testing.
