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
    batch_response = None
    error = None
    queries_raw = ""
    query = ""

    if request.method == "POST":
        queries_raw = request.form.get("queries", "").strip()
        if not queries_raw:
            queries_raw = request.form.get("query", "").strip()
            
        if queries_raw:
            lines = [q.strip() for q in queries_raw.split("\n") if q.strip()]
            
            if not lines:
                error = "Please enter at least one query."
            elif len(lines) == 1:
                query = lines[0]
                try:
                    response = requests.post(API_URL, json={"query": query})
                    if response.status_code == 429:
                        detail = response.json().get("detail", "Rate limit exceeded.")
                        error = f"Rate Limit Exceeded: {detail}"
                    else:
                        response.raise_for_status()
                        prediction = response.json()
                except requests.exceptions.RequestException as e:
                    error = f"Error connecting to the API: {e}"
            else:
                try:
                    response = requests.post(f"{BASE_API_URL}/predict/batch", json={"queries": lines})
                    
                    if response.status_code == 429:
                        detail = response.json().get("detail", "Rate limit or concurrency limit exceeded.")
                        error = f"Rate Limit / Concurrency Limit Exceeded: {detail}"
                    elif response.status_code == 504:
                        error = "Gateway Timeout: Inference took too long. Please try again with a smaller batch."
                    elif response.status_code == 422:
                        detail = response.json().get("detail", "Validation error.")
                        if isinstance(detail, list):
                            msgs = []
                            for d in detail:
                                loc = d.get("loc", [])
                                msg = d.get("msg", "")
                                msgs.append(f"{'.'.join(str(l) for l in loc)}: {msg}")
                            msg_str = "; ".join(msgs)
                        else:
                            msg_str = str(detail)
                        error = f"Validation Error: {msg_str}"
                    else:
                        response.raise_for_status()
                        batch_response = response.json()
                except requests.exceptions.RequestException as e:
                    error = f"Error connecting to the API: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        batch_response=batch_response,
        error=error,
        queries_raw=queries_raw,
        query=query
    )

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
