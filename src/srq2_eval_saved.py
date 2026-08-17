import json
import sys
from pathlib import Path
import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T
from srq2_evaluate import (load_packed_text, load_unpacked_text_for,
                           macro_f1_over, predict, HEADLINE_CATS)

ROOT = Path(__file__).resolve().parent.parent
PACKED_FEATURES = ROOT / "data" / "features_packed.jsonl"
RESULTS = ROOT / "results" / "srq2_eval_results.json"
LABEL2ID = T.LABEL2ID
CATEGORIES = T.CATEGORIES


def main():
    arm = sys.argv[1]
    save_dir = ROOT / "models" / f"{arm}_arm_saved"
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    model = AutoModelForSequenceClassification.from_pretrained(save_dir)

    packed_shas = [json.loads(l)["sha256"] for l in open(PACKED_FEATURES)]
    cat_by_sha = {json.loads(l)["sha256"]: json.loads(l)["category"]
                  for l in open(PACKED_FEATURES)}

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
    all_results = json.load(open(RESULTS)) if RESULTS.exists() else {}
    all_results[arm] = result
    with open(RESULTS, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"{arm}: unpacked {unp_macro:.4f}  packed {pk_macro:.4f}  delta {pk_macro-unp_macro:+.4f}")
    print(f"added to {RESULTS}")


if __name__ == "__main__":
    main()
