import os
from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# FastAPI endpoint (configurable via environment variables for Render/AWS)
API_URL = os.environ.get("API_URL", "http://localhost:8000/predict")

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

if __name__ == "__main__":
    app.run(port=5000, debug=True)
