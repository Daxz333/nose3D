"""Standalone head web server; dense inference uses the separate HRN environment."""
from flask import Flask, redirect, send_from_directory
from head_reconstruction.api import ROOT, head_api

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 13 * 1024 * 1024
app.register_blueprint(head_api)


@app.get("/")
def index():
    return redirect("/app/#/head")


@app.get("/app/")
def frontend():
    return send_from_directory(ROOT / "frontend" / "dist", "index.html")


@app.get("/app/<path:filename>")
def assets(filename):
    return send_from_directory(ROOT / "frontend" / "dist", filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=6007, threaded=True)
