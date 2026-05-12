"""
train.py
---------
Fine-tunes DualHeadBertClassifier on the preprocessed customer support dataset.

Steps:
  1. Load train/val CSVs
  2. Build PyTorch Datasets & DataLoaders
  3. Initialise model, AdamW optimiser, linear warmup scheduler
  4. Training loop with per-epoch evaluation
  5. Save best checkpoint (by val F1-macro) to model/saved_model/

Run:
    python training/scripts/train.py

GPU is strongly recommended. Falls back to CPU if no CUDA device is found.
"""

import sys
import random
import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report
)

# ── Make sure project root is on the path ─────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.bert_classifier import DualHeadBertClassifier

# ── Load Config ───────────────────────────────────────────────────────────────
CONFIG_PATH = ROOT / "training" / "configs" / "train_config.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = cfg["training"]["seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class CustomerSupportDataset(Dataset):
    def __init__(self, csv_path: Path, tokenizer: BertTokenizer, max_len: int):
        df = pd.read_csv(csv_path)
        self.texts           = df["text"].tolist()
        self.intent_labels   = df["intent_label"].tolist()
        self.priority_labels = df["priority_label"].tolist()
        self.tokenizer       = tokenizer
        self.max_len         = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "intent_label":   torch.tensor(self.intent_labels[idx],   dtype=torch.long),
            "priority_label": torch.tensor(self.priority_labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    all_intent_preds, all_intent_labels     = [], []
    all_priority_preds, all_priority_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_labels  = batch["intent_label"].to(device)
            priority_labels= batch["priority_label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = model.compute_loss(
                outputs["intent_logits"],   outputs["priority_logits"],
                intent_labels,              priority_labels,
            )
            total_loss += loss.item()

            all_intent_preds.extend(outputs["intent_logits"].argmax(dim=-1).cpu().tolist())
            all_intent_labels.extend(intent_labels.cpu().tolist())
            all_priority_preds.extend(outputs["priority_logits"].argmax(dim=-1).cpu().tolist())
            all_priority_labels.extend(priority_labels.cpu().tolist())

    intent_acc  = accuracy_score(all_intent_labels,   all_intent_preds)
    priority_acc= accuracy_score(all_priority_labels, all_priority_preds)

    _, _, intent_f1, _   = precision_recall_fscore_support(
        all_intent_labels, all_intent_preds, average="macro", zero_division=0
    )
    _, _, priority_f1, _ = precision_recall_fscore_support(
        all_priority_labels, all_priority_preds, average="macro", zero_division=0
    )

    return {
        "loss":         total_loss / len(loader),
        "intent_acc":   intent_acc,
        "priority_acc": priority_acc,
        "intent_f1":    intent_f1,
        "priority_f1":  priority_f1,
        "avg_f1":       (intent_f1 + priority_f1) / 2,
        "intent_report":  classification_report(
            all_intent_labels, all_intent_preds,
            target_names=["complaint", "inquiry", "feedback"], zero_division=0
        ),
        "priority_report": classification_report(
            all_priority_labels, all_priority_preds,
            target_names=["high", "medium", "low"], zero_division=0
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # Paths
    ckpt_dir       = ROOT / cfg["output"]["checkpoint_dir"]
    best_model_dir = ROOT / cfg["output"]["best_model_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer & datasets
    model_name = cfg["model"]["name"]
    max_len    = cfg["model"]["max_seq_length"]
    tokenizer  = BertTokenizer.from_pretrained(model_name)

    train_ds = CustomerSupportDataset(ROOT / cfg["data"]["train_csv"], tokenizer, max_len)
    val_ds   = CustomerSupportDataset(ROOT / cfg["data"]["val_csv"],   tokenizer, max_len)
    test_ds  = CustomerSupportDataset(ROOT / cfg["data"]["test_csv"],  tokenizer, max_len)

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = DualHeadBertClassifier(
        model_name=model_name,
        num_intent_labels=cfg["model"]["num_intent_labels"],
        num_priority_labels=cfg["model"]["num_priority_labels"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    # Optimiser & scheduler
    epochs       = cfg["training"]["epochs"]
    lr           = cfg["training"]["learning_rate"]
    warmup_ratio = cfg["training"]["warmup_ratio"]
    total_steps  = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    optimizer = AdamW(
        model.parameters(), lr=lr, weight_decay=cfg["training"]["weight_decay"]
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_f1  = 0.0
    history      = []
    log_every    = cfg["logging"]["log_every_n_steps"]
    grad_clip    = cfg["training"]["gradient_clip"]

    print(f"\n[train] Starting training — {epochs} epochs, {total_steps:,} total steps\n")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        step_count   = 0

        for step, batch in enumerate(train_loader, 1):
            input_ids       = batch["input_ids"].to(device)
            attention_mask  = batch["attention_mask"].to(device)
            intent_labels   = batch["intent_label"].to(device)
            priority_labels = batch["priority_label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = model.compute_loss(
                outputs["intent_logits"], outputs["priority_logits"],
                intent_labels, priority_labels,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            step_count   += 1

            if step % log_every == 0:
                avg = running_loss / step_count
                print(f"  Epoch {epoch} | Step {step:>5}/{len(train_loader)} | Loss: {avg:.4f}")

        # ── Validation ────────────────────────────────────────────────────────
        val_metrics = evaluate(model, val_loader, device)
        print(
            f"\nEpoch {epoch} ▸ "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Intent Acc: {val_metrics['intent_acc']:.4f} | "
            f"Priority Acc: {val_metrics['priority_acc']:.4f} | "
            f"Intent F1: {val_metrics['intent_f1']:.4f} | "
            f"Priority F1: {val_metrics['priority_f1']:.4f} | "
            f"Avg F1: {val_metrics['avg_f1']:.4f}\n"
        )

        # Save checkpoint every epoch
        ckpt_path = ckpt_dir / f"checkpoint_epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)

        # Save best model
        if val_metrics["avg_f1"] > best_val_f1:
            best_val_f1 = val_metrics["avg_f1"]
            torch.save(model.state_dict(), best_model_dir / "model_weights.pt")
            tokenizer.save_pretrained(str(best_model_dir))
            print(f"  ✅ New best model saved (avg F1: {best_val_f1:.4f})")

        history.append({"epoch": epoch, **{k: v for k, v in val_metrics.items() if k != "intent_report" and k != "priority_report"}})

    # ── Final Test Evaluation ─────────────────────────────────────────────────
    model.load_state_dict(torch.load(best_model_dir / "model_weights.pt", map_location=device))
    test_metrics = evaluate(model, test_loader, device)

    print("\n" + "═" * 60)
    print("FINAL TEST RESULTS")
    print("═" * 60)
    print(f"Intent   Accuracy: {test_metrics['intent_acc']:.4f}  F1: {test_metrics['intent_f1']:.4f}")
    print(f"Priority Accuracy: {test_metrics['priority_acc']:.4f}  F1: {test_metrics['priority_f1']:.4f}")
    print(f"Average  F1:       {test_metrics['avg_f1']:.4f}")
    print("\n── Intent Classification Report ──")
    print(test_metrics["intent_report"])
    print("── Priority Classification Report ──")
    print(test_metrics["priority_report"])

    # Save training history
    history_path = ckpt_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")
    print("✅ Training complete!")


if __name__ == "__main__":
    main()
