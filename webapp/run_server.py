"""
Production-ish launcher for the langvio web server.

app.py builds the langvio pipeline (Qwen-VL for both vision grounding and
language, over the hosted API or with local weights) at import time. This
launcher imports the app once and serves it with Flask's debug mode off.

Usage:
    cd webapp && python3 run_server.py
"""

import app  # noqa: E402  -- importing builds the pipeline once (module-level)

if __name__ == "__main__":
    if app.pipeline is None:
        raise SystemExit(
            "Failed to initialize the langvio pipeline. Set OPENROUTER_API_KEY "
            "in .env for the hosted Qwen API, or pip install 'langvio[vlm]' to "
            "run Qwen-VL locally."
        )
    app.app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)
