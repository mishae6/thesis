import json
import sys
from pathlib import Path

import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, set_seed)

# Make enrich.py importable regardless of where we run from
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from enrich import format_features, _cache_key

ROOT = SRC_DIR.parent                      # thesis/
FEATURES = ROOT / "data" / "features_full.jsonl"
CACHE = ROOT / "cache" / "descriptions"
CAPA_CACHE = ROOT / "cache" / "capa"

# Which arm to run: "llm" uses the cached LLM descriptions;
# "unenriched" uses the raw formatted feature text;
# "capa" uses the cached CAPA capability descriptions.
ARM = "capa"

CHECKPOINT = "answerdotai/ModernBERT-base"

CATEGORIES = ["backdoor", "downloader", "dropper", "informationstealer",
              "ransomware", "trojan", "virus", "worm"]
LABEL2ID = {c: i for i, c in enumerate(CATEGORIES)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}


def load_pairs():
    """Return a list of dicts: {sha256, description, category, key}.

    ARM controls the input representation:
      "llm"        -> the cached LLM behavioural description
      "unenriched" -> the raw formatted feature text (format_features)
      "capa"       -> the cached CAPA capability description
    """
    pairs = []
    missing = []
    for line in open(FEATURES):
        rec = json.loads(line)
        key = _cache_key(format_features(rec))

        if ARM == "unenriched":
            text = format_features(rec)
        elif ARM == "capa":
            cache_file = CAPA_CACHE / f"{key}.txt"
            if cache_file.exists():
                text = cache_file.read_text()
            else:
                # 4 feature-texts CAPA could not analyse: treat as empty
                # capability result (behaviourally like a packed sample).
                text = ""
        else:
            cache_file = CACHE / f"{key}.txt"
            if not cache_file.exists():
                missing.append(rec["sha256"][:12])
                continue
            text = cache_file.read_text()

        pairs.append({
            "sha256": rec["sha256"],
            "description": text,
            "category": rec["category"],
            "key": key,
        })
    if missing:
        raise RuntimeError(f"No cached description for: {missing}")
    return pairs


def make_splits(pairs, seed=42):
    """Group-aware stratified split into train/val/test (70/15/15).

    Samples sharing an identical feature-text (same 'key') are kept together
    in the same split, so no feature-text appears in both train and test.
    Multiple candidate splits are searched (across several seeds and folds);
    the one whose per-category test proportions are closest to 15% is chosen,
    to reduce lumpiness in categories with few unique feature-texts.
    """
    import numpy as np

    texts = [p["description"] for p in pairs]
    labels = [LABEL2ID[p["category"]] for p in pairs]
    groups = [p["key"] for p in pairs]

    n_cats = len(CATEGORIES)
    cat_totals = np.bincount(labels, minlength=n_cats)

    def imbalance(idx, totals, target):
        counts = np.bincount([labels[i] for i in idx], minlength=n_cats)
        share = counts / totals
        return np.abs(share - target).sum()

    # First split: search across several seeds x 7 folds for the test fold
    # whose per-category proportions are closest to 15%.
    best = None
    for s in range(seed, seed + 20):
        sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=s)
        for train_val_idx, test_idx in sgkf.split(texts, labels, groups):
            score = imbalance(test_idx, cat_totals, 0.15)
            if best is None or score < best[0]:
                best = (score, train_val_idx, test_idx)
    _, train_val_idx, test_idx = best

    tv_texts = [texts[i] for i in train_val_idx]
    tv_labels = [labels[i] for i in train_val_idx]
    tv_groups = [groups[i] for i in train_val_idx]
    tv_totals = np.bincount(tv_labels, minlength=n_cats)

    def imbalance_val(idx):
        counts = np.bincount([tv_labels[i] for i in idx], minlength=n_cats)
        share = counts / tv_totals
        return np.abs(share - (0.15 / 0.85)).sum()

    # Second split: same search for the validation fold from the remainder.
    best2 = None
    for s in range(seed, seed + 20):
        sgkf2 = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=s)
        for train_idx, val_idx in sgkf2.split(tv_texts, tv_labels, tv_groups):
            score = imbalance_val(val_idx)
            if best2 is None or score < best2[0]:
                best2 = (score, train_idx, val_idx)
    _, train_idx, val_idx = best2

    train_texts = [tv_texts[i] for i in train_idx]
    train_labels = [tv_labels[i] for i in train_idx]
    val_texts = [tv_texts[i] for i in val_idx]
    val_labels = [tv_labels[i] for i in val_idx]
    test_texts = [texts[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    # Self-verifying safeguard: no feature-text may span two splits.
    train_groups = {tv_groups[i] for i in train_idx}
    val_groups = {tv_groups[i] for i in val_idx}
    test_groups = {groups[i] for i in test_idx}
    assert not (train_groups & test_groups), "LEAK: feature-text in both train and test"
    assert not (train_groups & val_groups), "LEAK: feature-text in both train and val"
    assert not (val_groups & test_groups), "LEAK: feature-text in both val and test"

    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, test_idx


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


def compute_class_weights(train_labels):
    """Inverse-frequency weights so rare categories count more in the loss."""
    counts = torch.bincount(torch.tensor(train_labels), minlength=len(CATEGORIES))
    weights = len(train_labels) / (len(CATEGORIES) * counts.float())
    return weights


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy loss."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
    set_seed(42)
    print(f"ARM = {ARM}")
    pairs = load_pairs()
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, _ = make_splits(pairs)
    print(f"train: {len(train_texts)}  val: {len(val_texts)}  test: {len(test_texts)}")

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    train_ds = DescriptionDataset(train_texts, train_labels, tokenizer)
    val_ds = DescriptionDataset(val_texts, val_labels, tokenizer)
    test_ds = DescriptionDataset(test_texts, test_labels, tokenizer)

    class_weights = compute_class_weights(train_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT,
        num_labels=len(CATEGORIES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=str(ROOT / "models" / f"{ARM}_arm"),
        num_train_epochs=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        seed=42,
        report_to="none",
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print(f"\n--- final evaluation on TEST set (ARM={ARM}) ---")
    results = trainer.evaluate(test_ds)
    for k, v in sorted(results.items()):
        if k.startswith("eval_f1_") or k in ("eval_macro_f1", "eval_weighted_f1"):
            print(f"{k}: {v:.3f}")