# SRQ1 results — in-distribution three-arm comparison

SRQ1 asks how LLM-derived enrichment compares with CAPA-based (rule-based) enrichment and unenriched features for 8-category BODMAS malware classification, measured by macro-F1.

Each arm was produced by the same training procedure (`src/train.py`), run once per arm with only the input representation changed. All three arms share the same 9,207 samples, the same labels, the same group-aware train/validation/test split (computed from the feature-text hash, so it is identical across arms), the same model (ModernBERT-base), and the same configuration (4 epochs, batch size 8, learning rate 5e-5, seed 42, class-weighted loss). Test-set macro-F1 is the primary metric; weighted-F1 and per-category F1 are also recorded.

## Results

Per-category and aggregate test-set F1 for the three arms. Macro-F1 is the primary metric.

| Category | Unenriched | LLM | CAPA |
| --- | ---: | ---: | ---: |
| backdoor | 0.951 | 0.954 | 0.961 |
| downloader | 0.890 | 0.747 | 0.783 |
| dropper | 0.778 | 0.042 | 0.122 |
| informationstealer | 0.281 | 0.222 | 0.348 |
| ransomware | 0.764 | 0.451 | 0.889 |
| trojan | 0.529 | 0.427 | 0.575 |
| virus | 1.000 | 0.952 | 0.968 |
| worm | 0.581 | 0.534 | 0.564 |
| **macro-F1** | **0.722** | **0.541** | **0.651** |
| weighted-F1 | 0.709 | 0.597 | 0.696 |

The per-arm evaluation outputs are also stored as plain text in `srq1_unenriched.txt`, `srq1_llm.txt`, and `srq1_capa.txt`.

## How these results were captured (provenance)

The three arms were trained in different sessions, and the way their results were captured differs. This is stated plainly so the record is honest.

- **CAPA arm (2026-08-10):** the final test-set evaluation was captured directly from the terminal output of the training run.
- **Unenriched and LLM arms (earlier sessions):** the terminal output of these runs was not saved to a file. The final test-set F1 values were read from the terminal as each run finished and recorded into the project write-up notes at that time. The values above and in the corresponding text files are those recorded values. They are transcribed from notes taken at run time, not from a saved terminal log.

Because the unenriched and LLM arms are not backed by a saved terminal log, they can be regenerated at any time by re-running `src/train.py` with the corresponding arm (see below). The code, seed, and cached enrichment that produced all three arms are in this repository.

## How the code maps to the arms

The shared training/evaluation pipeline is `src/train.py`:

- Uses `ModernBERT-base`, **4 epochs**, batch size **8**, learning rate **5e-5**, **seed 42**, **class-weighted** loss, and a **group-aware stratified split** (to prevent the same feature-text from spanning splits).
- Builds inputs from `data/features_full.jsonl` via `load_pairs()` and the `ARM` switch:
  - `ARM = "unenriched"`: input text is `format_features(rec)` (raw formatted PE-derived static features).
  - `ARM = "llm"`: input text is taken from the cached LLM descriptions in `cache/descriptions/`, keyed by `_cache_key(format_features(rec))`.
  - `ARM = "capa"`: input text is taken from cached CAPA capability descriptions in `cache/capa/`, keyed by `_cache_key(format_features(rec))`; missing CAPA parses are treated as empty descriptions.

## Reproducibility note

Training was run on Apple Silicon (MPS), which does not guarantee fully deterministic computation. Re-running an arm may therefore produce values that differ marginally from those recorded here (typically within a few tenths of a percentage point on macro-F1). The data split itself is deterministic and identical across all three arms.

## To reproduce an arm

1. Set the arm in `src/train.py` to `"unenriched"`, `"llm"`, or `"capa"`.
2. Run `src/train.py`. The final section prints the test-set macro-F1, weighted-F1, and per-category F1.

Enrichment is cached and reused, so no re-enrichment is needed:

- LLM descriptions: `cache/descriptions/`
- CAPA descriptions: `cache/capa/`

The `data/` and `cache/` directories are gitignored (they contain the BODMAS feature data and the enrichment caches) and are backed up separately, so they are not part of this repository.
