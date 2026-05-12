"""
predictor.py
-------------
Loads the fine-tuned DualHeadBertClassifier and runs inference on raw text.

Returns:
    {
      "intent":           "complaint" | "inquiry" | "feedback",
      "priority":         "high" | "medium" | "low",
      "intent_confidence":   float (0–1),
      "priority_confidence": float (0–1),
      "flagged":          bool   (True if either confidence < threshold)
    }

Usage:
    from model.predictor import Predictor
    predictor = Predictor()
    result = predictor.predict("My order never arrived and it's been 2 weeks!")
"""

import torch
import torch.nn.functional as F
import yaml
from pathlib import Path
from transformers import BertTokenizer

from model.bert_classifier import DualHeadBertClassifier

# ── Label maps (index → string) ───────────────────────────────────────────────
INTENT_LABELS   = {0: "complaint", 1: "inquiry", 2: "feedback"}
PRIORITY_LABELS = {0: "high",      1: "medium",  2: "low"}

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "training" / "configs" / "train_config.yaml"


class Predictor:
    def __init__(self, model_dir: str | Path | None = None):
        # Load config
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)

        self.max_seq_len  = cfg["model"]["max_seq_length"]
        self.threshold    = cfg["inference"]["confidence_threshold"]
        model_name        = cfg["model"]["name"]

        # Resolve model directory
        if model_dir is None:
            model_dir = BASE_DIR / cfg["output"]["best_model_dir"]
        self.model_dir = Path(model_dir)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Predictor] Using device: {self.device}")

        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # Model
        self.model = DualHeadBertClassifier(
            model_name=model_name,
            num_intent_labels=cfg["model"]["num_intent_labels"],
            num_priority_labels=cfg["model"]["num_priority_labels"],
            dropout=cfg["model"]["dropout"],
        )

        weights_path = self.model_dir / "model_weights.pt"
        if weights_path.exists():
            self.model.load_state_dict(
                torch.load(weights_path, map_location=self.device)
            )
            print(f"[Predictor] Loaded weights from {weights_path}")
        else:
            print(f"[Predictor] WARNING: No weights found at {weights_path}. Using random init.")

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        """Run inference on a single text query."""
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Intent
        intent_probs = F.softmax(outputs["intent_logits"], dim=-1)[0]
        intent_idx   = intent_probs.argmax().item()
        intent_conf  = intent_probs[intent_idx].item()

        # Priority
        priority_probs = F.softmax(outputs["priority_logits"], dim=-1)[0]
        priority_idx   = priority_probs.argmax().item()
        priority_conf  = priority_probs[priority_idx].item()

        flagged = (intent_conf < self.threshold) or (priority_conf < self.threshold)

        return {
            "intent":              INTENT_LABELS[intent_idx],
            "priority":            PRIORITY_LABELS[priority_idx],
            "intent_confidence":   round(intent_conf,   4),
            "priority_confidence": round(priority_conf, 4),
            "flagged":             flagged,
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Run inference on a list of text queries."""
        return [self.predict(t) for t in texts]
