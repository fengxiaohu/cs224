"""Module D basic: polysemy analysis with full GloVe."""

from __future__ import annotations

import re
import sys
from pathlib import Path

RESEARCH_DIR = Path(__file__).resolve().parents[1]
if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

from config import POLYSEMY_WORDS, PREPROCESS_PATTERN
from src.glove_utils import glove_most_similar, load_glove
from src.run_utils import make_run_dir, save_config, save_json
from src.shared_data import load_corpus_artifacts


def extract_contexts(corpus, word: str, window: int = 5, max_examples: int = 5) -> list[dict]:
    """Extract context snippets containing word from preprocessed corpus."""
    examples = []
    pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
    for doc_idx, doc in enumerate(corpus):
        text = " ".join(doc)
        for m in pattern.finditer(text):
            start = max(0, m.start() - 40)
            end = min(len(text), m.end() + 40)
            snippet = text[start:end]
            examples.append({"doc_idx": doc_idx, "snippet": snippet})
            if len(examples) >= max_examples:
                return examples
    return examples


def annotate_neighbors(neighbors: list[tuple[str, float]]) -> list[dict]:
    """Placeholder sense annotation — manual labels recorded in output."""
    return [{"word": w, "score": s, "sense": "TBD_manual"} for w, s in neighbors]


def main() -> Path:
    run_dir = make_run_dir("module_d")
    artifacts = load_corpus_artifacts(run_dir)
    wv = load_glove()

    results = {}
    for word in POLYSEMY_WORDS:
        top10 = glove_most_similar(wv, word, topn=10)
        top30 = glove_most_similar(wv, word, topn=30)
        contexts = extract_contexts(artifacts.corpus, word)
        in_subset = sum(
            1 for doc in artifacts.corpus for tok in doc if tok == word
        )
        entry = {
            "in_imdb_subset_count": in_subset,
            "top10": top10,
            "top30": top30,
            "top10_annotated": annotate_neighbors(top10),
            "top30_annotated": annotate_neighbors(top30),
            "context_examples": contexts,
            "data_limit": None,
        }
        if in_subset == 0:
            entry["data_limit"] = "word absent from IMDB subset"
        elif in_subset < 3:
            entry["data_limit"] = "very few occurrences; sense diversity may be limited"
        results[word] = entry

    # Apply sense annotations based on neighbor analysis
    sense_labels = {
        "bank": {
            "financial": ["financial", "banks", "banking", "credit", "lending", "mortgage", "deposit"],
            "river": ["river", "shore", "embankment", "riverside"],
        },
        "plant": {
            "botanical": ["plants", "tree", "flower", "vegetation", "garden", "species"],
            "factory": ["factory", "facility", "industrial", "refinery", "manufacturing"],
        },
        "tie": {
            "neckwear": ["necktie", "bow", "suit", "wear"],
            "draw_equal": ["tied", "draw", "scoreless", "knot"],
        },
        "bark": {
            "tree": ["tree", "trees", "woody", "trunk"],
            "dog": ["dog", "dogs", "howl", "yelp"],
        },
    }
    for word, senses in sense_labels.items():
        annotated = []
        for w, s in results[word]["top10"]:
            assigned = "other"
            for sense_name, keywords in senses.items():
                if any(kw in w.lower() for kw in keywords):
                    assigned = sense_name
                    break
            annotated.append({"word": w, "score": s, "sense": assigned})
        results[word]["top10_annotated"] = annotated
        senses_found = set(a["sense"] for a in annotated if a["sense"] != "other")
        if len(senses_found) < 2:
            limit = results[word].get("data_limit") or ""
            results[word]["data_limit"] = (
                limit + "; top-10 shows primarily one sense cluster"
            ).strip("; ")

    save_json(run_dir / "module_d_polysemy.json", results)
    save_config(
        run_dir,
        {
            "module": "D",
            "words": POLYSEMY_WORDS,
            "glove_model": "glove-wiki-gigaword-200",
            "data_hashes": artifacts.hashes,
        },
    )
    print(f"Module D complete: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
