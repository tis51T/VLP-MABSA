#!/usr/bin/env python3
"""
process_input_data.py

Lightweight data processing script that follows the README instructions.

Features implemented:
- Tokenize texts with NLTK (falls back to simple split if NLTK not available)
- Extract aspects using spaCy NER when available; otherwise use a heuristic
- Extract opinion words by matching a provided sentiment lexicon (one-word-per-line or JSON list)
- Save outputs as JSON files: aspects, opinions, tokenized texts, sentiments
- Simple sentiment inconsistency filter (drop records where text and image labels disagree)
- Placeholders/wrappers for image region feature extraction and ANP extraction (see docstrings)

Usage examples:
  python scripts/process_input_data.py --mode text --input data/raw_texts.jsonl --outdir data/processed --lexicon resources/senti_lexicon.txt
  python scripts/process_input_data.py --mode sentiment --input data/labels.json --outdir data/processed --drop-inconsistent

The script is intentionally conservative: it won't attempt to install heavy libraries.
If spaCy or Detectron2 are present, it will use them. Otherwise it falls back to simple heuristics.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import nltk
    from nltk.tokenize import word_tokenize
    nltk.data.find("tokenizers/punkt")
except Exception:
    word_tokenize = None

try:
    import spacy
    _has_spacy = True
    try:
        _nlp = spacy.load("en_core_web_sm")
    except Exception:
        # will try to download if missing
        try:
            from spacy.cli import download

            download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            _nlp = None
            _has_spacy = False
except Exception:
    spacy = None
    _has_spacy = False
    _nlp = None

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kw):
        return x


def tokenize(text: str) -> List[str]:
    """Tokenize text into a list of tokens.

    Prefer NLTK's word_tokenize when available, otherwise use a simple regex.
    """
    if not text:
        return []
    if word_tokenize is not None:
        try:
            return word_tokenize(text)
        except Exception:
            pass
    # fallback simple tokenization
    return re.findall(r"\w+|[^\w\s]", text)


def extract_aspects_spacy(tokens: List[str], text: str) -> List[Tuple[int, int]]:
    """Use spaCy NER to extract entity spans as aspects (returns token index spans).

    Returns list of [start, end] inclusive indices (matching README examples where [4,5] spans two tokens).
    """
    if not _has_spacy or _nlp is None:
        return []
    doc = _nlp(text)
    spans = []
    token_texts = tokens
    # Build token char offsets to map spaCy entity char spans to token index spans
    offsets = []  # list of (start_char, end_char) for each token
    cursor = 0
    text_lower = text
    for tok in token_texts:
        # find tok starting from cursor
        idx = text_lower.find(tok, cursor)
        if idx == -1:
            # fallback: approximate
            idx = cursor
        offsets.append((idx, idx + len(tok)))
        cursor = idx + len(tok)

    for ent in doc.ents:
        # map ent.start_char..ent.end_char to token indices
        start_i = None
        end_i = None
        for i, (s, e) in enumerate(offsets):
            if s <= ent.start_char < e or (ent.start_char <= s < ent.end_char):
                if start_i is None:
                    start_i = i
                end_i = i
        if start_i is not None and end_i is not None:
            spans.append((start_i, end_i))
    return spans


def extract_aspects_heuristic(tokens: List[str]) -> List[Tuple[int, int]]:
    """A simple heuristic: sequences of Titlecase tokens (length >=1) are aspects.

    This is a fallback when no NER is available. It returns token index spans.
    """
    spans = []
    start = None
    for i, t in enumerate(tokens):
        if t and t[0].isupper() and not t.isupper():
            if start is None:
                start = i
        else:
            if start is not None:
                spans.append((start, i - 1))
                start = None
    if start is not None:
        spans.append((start, len(tokens) - 1))
    return spans


def extract_opinions(tokens: List[str], lexicon: Optional[set]) -> List[Tuple[int, int]]:
    """Find tokens that match words in the lexicon.

    Returns list of (i,i) spans for matched tokens.
    """
    spans = []
    if not lexicon:
        return spans
    for i, t in enumerate(tokens):
        if t.lower() in lexicon:
            spans.append((i, i))
    return spans


def load_lexicon(path: str) -> set:
    """Load a lexicon file. Supported formats:
    - plain text: one word per line
    - JSON list: ["word1","word2",...]
    - JSON object/dict: keys are words
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = set()
    with open(path, "r", encoding="utf8") as f:
        text = f.read().strip()
        if not text:
            return data
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                for w in parsed:
                    data.add(str(w).lower())
            elif isinstance(parsed, dict):
                for w in parsed.keys():
                    data.add(str(w).lower())
        except Exception:
            # treat as plain newline-separated
            for line in text.splitlines():
                w = line.strip()
                if not w:
                    continue
                # support lines like 'word\t...'
                w = w.split()[0]
                data.add(w.lower())
    return data


def read_input_texts(path: str) -> Sequence[Dict]:
    """Read input texts. Supports JSONL (one object per line) or a JSON list or a CSV with id,text columns."""
    if path.lower().endswith(".jsonl") or path.lower().endswith(".ndjson"):
        out = []
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf8") as f:
            obj = json.load(f)
            if isinstance(obj, list):
                return obj
            # otherwise maybe map of id->text
            return [{"id": k, "text": v} for k, v in obj.items()]
    # minimal CSV support
    if path.lower().endswith(".csv"):
        import csv

        out = []
        with open(path, newline="", encoding="utf8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                out.append(row)
        return out
    raise ValueError("Unsupported input file format: %s" % path)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def process_texts(input_path: str, outdir: str, lexicon_path: Optional[str] = None):
    records = read_input_texts(input_path)
    lex = None
    if lexicon_path:
        lex = load_lexicon(lexicon_path)

    aspects = {}
    opinions = {}
    tokenized = {}

    for rec in tqdm(records):
        # expect each record to have 'id' and 'text' fields
        rid = rec.get("id") or rec.get("text_id") or rec.get("image_id") or rec.get("name")
        text = rec.get("text") or rec.get("caption") or rec.get("sentence") or ""
        if rid is None:
            # try to generate an id
            rid = str(hash(text))
        toks = tokenize(text)
        tokenized[rid] = toks

        # aspect extraction
        asp_spans = []
        if _has_spacy and _nlp is not None:
            asp_spans = extract_aspects_spacy(toks, text)
        else:
            asp_spans = extract_aspects_heuristic(toks)

        aspects[rid] = {
            "aspect_spans": [list(s) for s in asp_spans],
            "aspect_texts": [[toks[i] for i in range(s[0], s[1] + 1)] for s in asp_spans],
        }

        # opinions
        op_spans = extract_opinions(toks, lex)
        opinions[rid] = {
            "opinion_spans": [list(s) for s in op_spans],
            "opinion_texts": [[toks[i] for i in range(s[0], s[1] + 1)] for s in op_spans],
        }

    # save
    os.makedirs(outdir, exist_ok=True)
    save_json(aspects, os.path.join(outdir, "aspects.json"))
    save_json(opinions, os.path.join(outdir, "opinions.json"))
    save_json(tokenized, os.path.join(outdir, "tokenized.json"))
    print("Saved aspects/opinions/tokenized to", outdir)


def process_sentiments(input_path: str, outdir: str, drop_inconsistent: bool = False):
    """Process sentiment labels and optionally drop inconsistent text/image labels.

    The input is expected to contain records with 'id' and either 'sentiment' or
    separate 'text_sentiment' and 'image_sentiment' fields. Output is a mapping id->label.
    """
    recs = read_input_texts(input_path)
    out = {}
    for r in recs:
        rid = r.get("id") or r.get("text_id") or r.get("image_id")
        if rid is None:
            continue
        text_label = r.get("text_sentiment")
        image_label = r.get("image_sentiment")
        label = r.get("sentiment")
        if label is None:
            # if there are separate labels, decide
            if text_label is not None and image_label is not None:
                if text_label == image_label:
                    label = text_label
                else:
                    if drop_inconsistent:
                        # skip
                        continue
                    # fallback: pick text label
                    label = text_label
            else:
                label = text_label or image_label
        out[rid] = label
    os.makedirs(outdir, exist_ok=True)
    save_json(out, os.path.join(outdir, "sentiments.json"))
    print("Saved sentiments to", outdir)


def extract_region_features_placeholder(image_list: Sequence[str], out_dir: str, topk: int = 36):
    """Placeholder helper: instructs the user how to run a Faster-RCNN extractor.

    If torchvision or detectron2 is installed and you want, this function can be
    extended to run extraction here. For now it writes a small TODO file describing the command.
    """
    todo = {
        "note": "This is a placeholder. To extract Faster-RCNN region features, run a script using Detectron2 or the bottom-up-attention repo.",
        "suggested_topk": topk,
        "example_detectron2_call": "python detectron2_feature_extractor.py /path/to/images /path/to/outdir --topk %d" % topk,
    }
    os.makedirs(out_dir, exist_ok=True)
    save_json(todo, os.path.join(out_dir, "REGION_FEATURES_TODO.json"))
    print("Wrote REGION_FEATURES_TODO.json with instructions to", out_dir)


def extract_anp_placeholder(image_list: Sequence[str], out_dir: str):
    todo = {
        "note": "Use the DeepSentiBank ANP extractor externally. Provide a file with image paths and run the extractor."
    }
    os.makedirs(out_dir, exist_ok=True)
    save_json(todo, os.path.join(out_dir, "ANP_TODO.json"))
    print("Wrote ANP_TODO.json with instructions to", out_dir)


def main(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="Process raw dataset into pretrain-friendly files")
    p.add_argument("--mode", choices=["text", "images", "anp", "sentiment", "all"], default="all")
    p.add_argument("--input", required=True, help="Input file (json/jsonl/csv list of records)")
    p.add_argument("--outdir", required=True, help="Output directory to write processed files")
    p.add_argument("--lexicon", help="Path to sentiment lexicon (txt or json)")
    p.add_argument("--drop-inconsistent", action="store_true", help="Drop records where text and image sentiment disagree")
    p.add_argument("--topk", type=int, default=36, help="Top-K image regions to keep when extracting features")
    args = p.parse_args(argv)

    if args.mode in ("text", "all"):
        print("Processing text -> aspects/opinions...")
        process_texts(args.input, args.outdir, lexicon_path=args.lexicon)

    if args.mode in ("sentiment", "all"):
        print("Processing sentiment labels...")
        process_sentiments(args.input, args.outdir, drop_inconsistent=args.drop_inconsistent)

    if args.mode in ("images", "all"):
        print("Preparing image region feature extraction instructions...")
        # read input to gather images list if possible
        recs = []
        try:
            recs = read_input_texts(args.input)
        except Exception:
            recs = []
        image_list = []
        for r in recs:
            img = r.get("image") or r.get("image_path") or r.get("img_path")
            if img:
                image_list.append(img)
        extract_region_features_placeholder(image_list, args.outdir, topk=args.topk)

    if args.mode in ("anp", "all"):
        print("Preparing ANP extraction instructions...")
        recs = []
        try:
            recs = read_input_texts(args.input)
        except Exception:
            recs = []
        image_list = []
        for r in recs:
            img = r.get("image") or r.get("image_path") or r.get("img_path")
            if img:
                image_list.append(img)
        extract_anp_placeholder(image_list, args.outdir)


if __name__ == "__main__":
    main()
