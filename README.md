# AI-Based Customer Support Query Classification

Fine-tuned **BERT** (`bert-base-uncased`) that classifies customer support queries into:

| Output | Labels |
|--------|--------|
| **Intent** | `complaint` / `inquiry` / `feedback` |
| **Priority** | `high` / `medium` / `low` |

---

## 1. Project Structure

```text
MLops/
├── data/
│   ├── raw/customer_support_100k.csv    ← Real dataset (100k rows)
│   └── processed/                        ← Auto-generated splits
├── training/
│   ├── scripts/
│   │   ├── assign_priority_labels.py
│   │   ├── preprocess.py
│   │   └── train.py
│   └── configs/train_config.yaml
├── model/
│   ├── bert_classifier.py
│   ├── predictor.py
│   └── saved_model/                      ← Best checkpoint saved here
├── api/
│   ├── main.py
│   ├── routes/predict.py
│   └── schemas/query_schema.py
├── monitoring/
│   ├── logger.py
│   └── flagging.py
├── tests/
│   ├── test_model.py
│   └── test_api.py
├── logs/                                  ← Auto-created at runtime
└── requirements.txt
```

---

## 2. Setup Guide (For a Fresh PC)

Follow these exact steps to get the project running on a new machine from scratch.

### Step 1: Get the Code & Open Terminal
Clone or download this repository to your PC, then open a terminal (Command Prompt, PowerShell, or VS Code Terminal) inside the `MLops` folder.

### Step 2: Create a Virtual Environment (Recommended)
This keeps dependencies isolated. Run these commands:
```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate
# Activate on Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
Install all required libraries.
```bash
python -m pip install -r requirements.txt
```
*(Note: If you have an NVIDIA GPU, you should install PyTorch with CUDA support **before** running the above command using: `python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`)*

### Step 4: Add the Dataset
The script expects a file named `customer_support_100k.csv`. Make sure you place this file exactly here:
`MLops/data/raw/customer_support_100k.csv`

### Step 5: Process Data & Train the Model
Run these three commands one by one to label the data, prepare the splits, and train the AI model:
```bash
# 1. Generate Priority Labels
python training/scripts/assign_priority_labels.py

# 2. Clean Text & Split Data
python training/scripts/preprocess.py

# 3. Start Training (Grab a coffee, this takes a few minutes!)
python training/scripts/train.py
```
*Once finished, the trained weights are saved to `model/saved_model/`.*

### Step 6: Start the API
Now that the model is trained, spin up the server to test it!
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
> [!TIP]
> Open **[http://localhost:8000/docs](http://localhost:8000/docs)** in your browser to use the interactive testing UI!

---

## 3. Requirements

Below is the complete `requirements.txt` used for this project:

```text
# Core ML / NLP
torch>=2.1.0
transformers>=4.38.0
tokenizers>=0.15.0

# Data
pandas>=2.1.0
numpy>=1.26.0
scikit-learn>=1.4.0

# API
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
pydantic>=2.6.0

# Config
pyyaml>=6.0.1
python-dotenv>=1.0.1

# Testing
pytest>=8.0.0
httpx>=0.27.0          # required by FastAPI TestClient

# Utilities
tqdm>=4.66.0
```

---

## Additional Information

### API Usage Example (cURL)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"query": "My order arrived broken and I want a refund immediately!"}'
```

**Response:**
```json
{
  "query": "My order arrived broken and I want a refund immediately!",
  "intent": "complaint",
  "priority": "high",
  "intent_confidence": 0.9742,
  "priority_confidence": 0.9105,
  "flagged": false,
  "timestamp": "2025-05-11T14:30:00+00:00"
}
```

### Confidence Flagging
If the `intent_confidence` or `priority_confidence` score drops below the threshold (`0.70`), the response will indicate `"flagged": true`. These flagged predictions are saved automatically to `logs/flagged_predictions.jsonl` for human review and iterative model retraining.
