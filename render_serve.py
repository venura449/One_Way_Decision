"""
Render Web Service entrypoint — always listens on 0.0.0.0.

The dashboard Start Command often defaults to host 127.0.0.1, which makes
Render's port probe fail ("No open ports detected on 0.0.0.0"). Use:

    python render_serve.py
"""

import os

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
