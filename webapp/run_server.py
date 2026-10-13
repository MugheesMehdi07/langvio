"""
Production-ish launcher for the langvio web server.

app.py builds the langvio pipeline (local Qwen2.5-VL for both vision grounding
and language) at import time. Running app.py directly uses Flask's debug reloader, which spawns a
second process and would load the multi-GB VLM twice. This launcher imports the
app once and runs it with the reloader disabled.

Usage:
    cd webapp && python3 run_server.py
"""

import app  # noqa: E402  -- importing builds the pipeline once (module-level)

if __name__ == "__main__":
    if app.pipeline is None:
        raise SystemExit(
            "Failed to initialize the langvio pipeline. "
            "Check that the VLM backend is installed (pip install langvio[vlm])."
        )
    app.app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)
