# MrPerfectV5 — Ready for Git & Render

This repository is prepared for direct upload to GitHub (or other Git host) and deployment to Render.com.
It contains two intended services:
- **Web (UI)** — serves the Flask UI at `ui/app.py` (Gunicorn start command provided).
- **Background worker** — runs `ui/fetcher_ui_bridge.py` which auto-detects and runs your `fetcher.py` and writes `ui/predictions.log`.

## Quick local run
1. Create & activate a venv (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # on Windows: .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Start the bridge (in one terminal):
   ```bash
   python ui/fetcher_ui_bridge.py
   ```
3. Start the web UI (in another terminal):
   ```bash
   gunicorn ui.app:app --bind 0.0.0.0:8000
   ```
4. Open http://localhost:8000

## How to deploy to Render (Git-backed)
1. Create a new GitHub repository and push this project.
2. In Render dashboard create two services:
   - Web Service (UI): build `pip install -r requirements.txt`, start `gunicorn ui.app:app --bind 0.0.0.0:$PORT`
   - Background Worker: build `pip install -r requirements.txt`, start `python ui/fetcher_ui_bridge.py`
3. Add any environment variables (API keys, secrets) in Render's dashboard settings — do not commit secrets to Git.
4. Watch logs in Render to verify the worker started and the web service is healthy.

## Notes
- Files written to disk are ephemeral on Render. If you need persistent storage for logs or data, use Render Disks or external storage (S3).
- If your `fetcher.py` requires custom system packages, consider using a Docker image and deploying container-based services on Render.

---
Generated automatically. If you want me to directly push the repo to GitHub for you or create the Render services programmatically, tell me which GitHub repo name and whether you want me to include placeholders for secrets.
