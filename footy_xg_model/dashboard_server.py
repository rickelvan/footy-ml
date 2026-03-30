"""
Serve the ML dashboard over HTTP so the browser can trigger a full pipeline
retrain after uploading a new player-season CSV.

From the project root (the folder that contains ``footy_xg_model``):

    pip install flask
    python -m footy_xg_model.dashboard_server

Open http://127.0.0.1:8765/ — use the CSV drop zone to replace
``config.RAW_PLAYER_SEASON_PATH`` and run the full pipeline in a background
thread. Progress is polled from ``/api/status`` (see ``pipeline_progress``).

Opening ``ml_dashboard.html`` from disk cannot upload or retrain; the server is
required for that flow.
"""

from __future__ import annotations

import threading
import traceback
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

from . import config
from . import main as main_mod
from . import pipeline_progress

app = Flask(__name__)
_run_lock = threading.Lock()


def _inject_server_flag(html: str) -> str:
    """Inject API flag + CSS so CSV retrain shows even if inline JS errors later."""
    inj = (
        "<script>window.FOOTY_DASHBOARD_API=1;</script>"
        '<style id="footy-dash-server-css">'
        "#csvServerOffline{display:none!important}"
        "#csvServerZone{display:block!important}"
        "</style>"
    )
    lowered = html.lower()
    i = lowered.find("<body")
    if i == -1:
        return inj + html
    j = html.find(">", i) + 1
    return html[:j] + inj + html[j:]


@app.get("/")
def index() -> Response:
    path = config.DASHBOARD_PATH
    if not path.exists():
        return Response(
            "Run the pipeline once first: python -m footy_xg_model.main\n"
            f"(expected dashboard at {path})",
            status=404,
            mimetype="text/plain; charset=utf-8",
        )
    html = path.read_text(encoding="utf-8")
    return Response(
        _inject_server_flag(html),
        mimetype="text/html; charset=utf-8",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/players/<filename>")
def players_photo(filename: str):
    """Serve copied headshots from ``artifacts/players`` (same filenames as in the HTML)."""
    root = (Path(config.DATA_OUT_DIR) / "players").resolve()
    if not root.is_dir():
        return Response("Not found", status=404)
    path = (root / filename).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return Response("Not found", status=404)
    if not path.is_file():
        return Response("Not found", status=404)
    return send_from_directory(str(root), filename)


@app.get("/api/health")
def health():
    return jsonify({"ok": True})


@app.get("/api/status")
def status():
    resp = jsonify(pipeline_progress.snapshot())
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


@app.post("/api/retrain")
def retrain():
    if not _run_lock.acquire(blocking=False):
        return jsonify({"error": "A training run is already in progress."}), 409
    if "file" not in request.files:
        _run_lock.release()
        return jsonify({"error": "No file field 'file' in form."}), 400
    f = request.files["file"]
    if not f.filename or not f.filename.lower().endswith(".csv"):
        _run_lock.release()
        return jsonify({"error": "Please upload a .csv file."}), 400

    try:
        dest = Path(config.RAW_PLAYER_SEASON_PATH)
        dest.parent.mkdir(parents=True, exist_ok=True)
        f.save(str(dest))
    except OSError as e:
        _run_lock.release()
        return jsonify({"error": str(e)}), 500

    pipeline_progress.reset()
    pipeline_progress.start_run()

    def job() -> None:
        try:
            main_mod.run_pipeline(track_progress=True)
            pipeline_progress.set_done()
        except BaseException as e:
            tb = traceback.format_exc()
            print(f"[dashboard_server retrain] pipeline failed:\n{tb}", flush=True)
            pipeline_progress.set_error(f"{type(e).__name__}: {e}")
        finally:
            _run_lock.release()

    threading.Thread(target=job, daemon=True).start()
    return jsonify({"started": True})


def main() -> None:
    print("Footy ML dashboard server")
    print("  Open http://127.0.0.1:8765/ in your browser")
    print("  Drop a player-season CSV to retrain the full pipeline.")
    app.run(host="127.0.0.1", port=8765, threaded=True, debug=False)


if __name__ == "__main__":
    main()
