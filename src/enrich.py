import json
import ollama

MODEL = "mistral"

PROMPT_TEMPLATE = """You are analysing the static features of a Windows PE file to describe what the program appears capable of doing. You are given only the readable static features below (imported libraries and functions, section names, and per-section entropy). Describe the program's likely behaviour based strictly on this evidence.

Feature record:
{feature_text}

Instructions:
- Describe the behaviour and capabilities suggested by the imports, sections, and entropy values. Focus on general behaviour types where the evidence supports them: file operations, network activity, persistence, process or memory manipulation, and evasion or packing.
- Base every statement on the features listed. Do not infer capabilities that the features do not support, and do not invent behaviour that is not evidenced.
- Do not name, label, or imply a malware category (for example trojan, worm, ransomware, backdoor). Describe behaviour only.
- Tie each described behaviour to the observed imports or sections, written as natural prose rather than a list of citations.
- If the features are sparse and provide limited behavioural evidence, state that the indicators are limited rather than describing behaviour that is not present.
- Respond with a single short description of no more than 100 words. Give the description directly, with no preamble, no restating of these instructions, and no closing commentary."""


def format_features(record):
    lines = []

    lines.append(f"Number of sections: {record['num_sections']}")

    lines.append("Sections (name and entropy):")
    entropies = record["section_entropies"]
    for name in record["section_names"]:
        ent = entropies.get(name, "n/a")
        lines.append(f"  {name}: entropy {ent}")

    lines.append("Imported libraries and functions:")
    imports = record["imports"]
    if imports:
        for dll, funcs in imports.items():
            func_list = ", ".join(funcs) if funcs else "(no named functions)"
            lines.append(f"  {dll}: {func_list}")
    else:
        lines.append("  (no imports found)")

    return "\n".join(lines)


def enrich_one(record):
    feature_text = format_features(record)
    prompt = PROMPT_TEMPLATE.format(feature_text=feature_text)

    response = ollama.generate(
        model=MODEL,
        prompt=prompt,
        options={"temperature": 0.1},
    )

    return response["response"].strip()
    
from pathlib import Path
import hashlib

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "descriptions"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(feature_text):
    return hashlib.sha256(feature_text.encode()).hexdigest()


def enrich_one_cached(record):
    feature_text = format_features(record)
    key = _cache_key(feature_text)
    cache_file = CACHE_DIR / f"{key}.txt"

    if cache_file.exists():
        return cache_file.read_text()

    description = enrich_one(record)
    cache_file.write_text(description)
    return description