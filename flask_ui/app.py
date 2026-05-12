import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import requests

app = Flask(__name__)
app.secret_key = "super_secret_key_for_flash" # Needed for flash messages

# FastAPI endpoint (configurable via environment variables for Render/AWS)
API_URL = os.environ.get("API_URL", "http://localhost:8000/predict")
BASE_API_URL = API_URL.replace("/predict", "").replace("/invocations", "")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    query = ""

    if request.method == "POST":
        query = request.form.get("query", "")
        if query:
            try:
                response = requests.post(API_URL, json={"query": query})
                response.raise_for_status()
                prediction = response.json()
            except requests.exceptions.RequestException as e:
                error = f"Error connecting to the API: {e}"

    return render_template("index.html", prediction=prediction, error=error, query=query)

@app.route("/dashboard")
def dashboard():
    analytics_data = {"complaints": 0, "high_priority": 0, "total": 0}
    try:
        res = requests.get(f"{BASE_API_URL}/analytics")
        if res.ok:
            analytics_data = res.json()
    except Exception as e:
        flash(f"Error fetching analytics: {e}", "error")
        
    return render_template("dashboard.html", data=analytics_data)

@app.route("/feedback", methods=["POST"])
def feedback():
    query = request.form.get("query")
    intent = request.form.get("expected_intent")
    priority = request.form.get("expected_priority")
    
    try:
        res = requests.post(f"{BASE_API_URL}/feedback", json={
            "query": query, 
            "expected_intent": intent, 
            "expected_priority": priority
        })
        if res.ok:
            flash("Feedback recorded! The system will learn from this.", "success")
    except Exception as e:
        flash("Failed to submit feedback.", "error")
        
    return redirect(url_for("index"))

@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        res = requests.post(f"{BASE_API_URL}/retrain")
        if res.ok:
            flash(res.json().get("message", "Retraining started."), "success")
        else:
            flash("Failed to start retraining.", "error")
    except Exception as e:
        flash("Failed to contact API for retraining.", "error")
        
    return redirect(url_for("dashboard"))

if __name__ == "__main__":
    app.run(port=5000, debug=True)
