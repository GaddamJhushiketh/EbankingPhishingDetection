"""Main application routes."""

from flask import Blueprint, jsonify, render_template


main_blueprint = Blueprint("main", __name__)


@main_blueprint.get("/")
def index():
    """Render the project landing page."""

    return render_template("index.html")


@main_blueprint.get("/health")
def health():
    """Return a lightweight application health response."""

    return jsonify({"status": "ok", "service": "ebanking-phishing-detection"})
