

import json
import sys
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, set_seed)

# Make enrich.py importable regardless of where we run from
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from enrich import format_features, _cache_key

ROOT = SRC_DIR.parent                      # thesis/
FEATURES = ROOT / "data" / "features_poc.jsonl"
CACHE = ROOT / "cache" / "descriptions"

CHECKPOINT = "answerdotai/ModernBERT-base"

CATEGORIES = ["backdoor", "downloader", "dropper", "informationstealer",
              "ransomware", "trojan", "virus", "worm"]
LABEL2ID = {c: i for i, c in enumerate(CATEGORIES)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}


def load_pairs():
    """Return a list of dicts: {sha256, description, category}."""
    pairs = []
    missing = []
    for line in open(FEATURES):
        rec = json.loads(line)
        key = _cache_key(format_features(rec))
        cache_file = CACHE / f"{key}.txt"
        if not cache_file.exists():
            missing.append(rec["sha256"][:12])
            continue
        pairs.append({
            "sha256": rec["sha256"],
            "description": cache_file.read_text(),
            "category": rec["category"],
        })
    if missing:
        raise RuntimeError(f"No cached description for: {missing}")
    return pairs


def make_splits(pairs, seed=42):
    """Stratified split: 2 train / 1 eval per category (16/8)."""
    texts = [p["description"] for p in pairs]
    labels = [LABEL2ID[p["category"]] for p in pairs]
    return train_test_split(
        texts, labels,
        test_size=1 / 3,
        stratify=labels,
        random_state=seed,
    )


def tokenize(texts, tokenizer):
    return tokenizer(
        texts,
        truncation=True,
        max_length=512,
        padding=True,
        return_tensors="pt",
    )

class DescriptionDataset(torch.utils.data.Dataset):
    """Wraps tokenised texts + labels in the format Trainer expects."""

    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenize(texts, tokenizer)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    per_class = f1_score(labels, preds, average=None,
                         labels=list(range(len(CATEGORIES))))
    metrics = {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }
    for cat, score in zip(CATEGORIES, per_class):
        metrics[f"f1_{cat}"] = score
    return metrics

if __name__ == "__main__":
    set_seed(42)
    pairs = load_pairs()
    train_texts, eval_texts, train_labels, eval_labels = make_splits(pairs)
    print(f"train: {len(train_texts)}  eval: {len(eval_texts)}")

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    train_ds = DescriptionDataset(train_texts, train_labels, tokenizer)
    eval_ds = DescriptionDataset(eval_texts, eval_labels, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT,
        num_labels=len(CATEGORIES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=str(ROOT / "models" / "poc"),
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        seed=42,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    
    results = trainer.evaluate()
    print("\n--- final evaluation ---")
    for k, v in sorted(results.items()):
        if k.startswith("eval_f1_") or k in ("eval_macro_f1", "eval_weighted_f1"):
            print(f"{k}: {v:.3f}")