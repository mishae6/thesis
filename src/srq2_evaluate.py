import json
import sys
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, set_seed)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T
from enrich import format_features

ROOT = Path(__file__).resolve().parent.parent
PACKED_FEATURES = ROOT / "data" / "features_packed.jsonl"
PACKED_LLM = ROOT / "results" / "srq2_llm_packed.jsonl"
PACKED_CAPA = ROOT / "results" / "srq2_capa_packed.jsonl"
RESULTS = ROOT / "results" / "srq2_eval_results.json"

CATEGORIES = T.CATEGORIES
LABEL2ID = T.LABEL2ID
HEADLINE_CATS = ["backdoor", "downloader", "informationstealer",
                 "ransomware", "trojan", "worm"]


def macro_f1_over(labels, preds, cats):
    idxs = [LABEL2ID[c] for c in cats]
    return f1_score(labels, preds, average="macro", labels=idxs)


def load_packed_text(arm):
    if arm == "unenriched":
        return {json.loads(l)["sha256"]: format_features(json.loads(l))
                for l in open(PACKED_FEATURES)}
    if arm == "llm":
        return {json.loads(l)["sha256"]: json.loads(l)["description"]
                for l in open(PACKED_LLM)}
    if arm == "capa":
        return {json.loads(l)["sha256"]: json.loads(l)["description"]
                for l in open(PACKED_CAPA)}


def load_unpacked_text_for(shas, arm):
    T.ARM = arm
    pairs = T.load_pairs()
    by_sha = {p["sha256"]: p["description"] for p in pairs}
    return {s: by_sha[s] for s in shas if s in by_sha}


def predict(model, tokenizer, texts):
    ds = T.DescriptionDataset(texts, [0] * len(texts), tokenizer)
    dl = torch.utils.data.DataLoader(ds, batch_size=8)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device).eval()
    preds = []
    with torch.no_grad():
        for batch in dl:
            inp = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            preds.extend(model(**inp).logits.argmax(-1).cpu().numpy().tolist())
    return preds


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else None
    if arm not in ("unenriched", "llm", "capa"):
        sys.exit("usage: python src/srq2_evaluate.py [unenriched|llm|capa]")

    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(T.CHECKPOINT)

    packed_shas = [json.loads(l)["sha256"] for l in open(PACKED_FEATURES)]
    cat_by_sha = {json.loads(l)["sha256"]: json.loads(l)["category"]
                  for l in open(PACKED_FEATURES)}

    print(f"\n===== ARM: {arm} =====")
    T.ARM = arm
    pairs = T.load_pairs()
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, _ = T.make_splits(pairs)

    train_ds = T.DescriptionDataset(train_texts, train_labels, tokenizer)
    val_ds = T.DescriptionDataset(val_texts, val_labels, tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        T.CHECKPOINT, num_labels=len(CATEGORIES),
        id2label=T.ID2LABEL, label2id=LABEL2ID)
    class_weights = T.compute_class_weights(train_labels)
    args = TrainingArguments(
        output_dir=str(ROOT / "models" / f"{arm}_arm"),
        num_train_epochs=4, per_device_train_batch_size=8,
        per_device_eval_batch_size=8, learning_rate=5e-5,
        eval_strategy="epoch", logging_strategy="epoch",
        save_strategy="no", seed=42, report_to="none")
    trainer = T.WeightedTrainer(
        class_weights=class_weights, model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=T.compute_metrics)
    trainer.train()

    save_dir = ROOT / "models" / f"{arm}_arm_saved"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"saved model to {save_dir}")

    unpacked_text = load_unpacked_text_for(packed_shas, arm)
    packed_text = load_packed_text(arm)
    eval_shas = [s for s in packed_shas if s in unpacked_text and s in packed_text]
    eval_labels = [LABEL2ID[cat_by_sha[s]] for s in eval_shas]
    unp_preds = predict(model, tokenizer, [unpacked_text[s] for s in eval_shas])
    pk_preds = predict(model, tokenizer, [packed_text[s] for s in eval_shas])

    unp_macro = macro_f1_over(eval_labels, unp_preds, HEADLINE_CATS)
    pk_macro = macro_f1_over(eval_labels, pk_preds, HEADLINE_CATS)
    per_cat = {}
    for c in CATEGORIES:
        idx = [LABEL2ID[c]]
        per_cat[c] = {
            "unpacked": round(f1_score(eval_labels, unp_preds, average="macro", labels=idx), 4),
            "packed": round(f1_score(eval_labels, pk_preds, average="macro", labels=idx), 4),
        }

    result = {
        "n_eval": len(eval_shas),
        "headline_macro_unpacked": round(unp_macro, 4),
        "headline_macro_packed": round(pk_macro, 4),
        "headline_delta": round(pk_macro - unp_macro, 4),
        "per_category": per_cat,
    }

    all_results = {}
    if RESULTS.exists():
        all_results = json.load(open(RESULTS))
    all_results[arm] = result
    with open(RESULTS, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{arm}: unpacked macro-F1 {unp_macro:.4f}  packed {pk_macro:.4f}  delta {pk_macro-unp_macro:+.4f}")
    print(f"written/updated: {RESULTS}")


if __name__ == "__main__":
    main()
