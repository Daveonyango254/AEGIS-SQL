"""Build the Spanish language-ablation mirror of the BIRD dev sample.

Creates ``data/bird_es/`` — a parallel BIRD data dir in which the questions of
the SAME seed-42 stratified 100-query sample are machine-translated to Spanish
while everything else (question ids, order, evidence, gold SQL, difficulty,
databases, schemas) is untouched. The whole ablation then becomes a flag on the
unchanged pipeline:

    python scripts/make_spanish_sample.py                       # one-time build
    python run_full_evaluation.py --config config.yaml --seed 42 \
           --num_queries 100 --stratify --bird_path data/bird_es \
           --output_name local_100_es

Design guarantees:
* **Identical sample.** The script derives the sample by calling the *same*
  ``BIRDLoader.load_queries`` + ``evaluation.sampling.sample_queries`` the eval
  driver uses — same code, same seed, same dev.json order ⇒ same 100 ids.
* **Single variable.** Only the ``question`` field of the sampled records is
  replaced. Evidence stays English (it quotes exact column names/literals), and
  the translator is instructed to keep quoted strings, named entities, numbers,
  codes, and acronyms VERBATIM — database values are English, so translating
  'Prague' to 'Praga' would break value grounding and measure translation
  damage instead of multilingual capability.
* **Resumable + reviewable.** Each translation is checkpointed to
  ``<out>/translations.jsonl`` (``{question_id, en, es}``); reruns skip done
  ids, and the file doubles as the human-review / paper-appendix artifact.

Requires OPENAI_API_KEY (translation via the configured remote model,
default gpt-4.1-mini, temperature 0). ~100 short calls ≈ well under $0.01.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:  # pragma: no cover
    from loguru import logger
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("make_spanish_sample")
    logging.basicConfig(level=logging.INFO)

TRANSLATOR_SYSTEM = (
    "You are a professional English-to-Spanish translator working on a database "
    "benchmark. Translate the user's question into natural, fluent Spanish.\n"
    "STRICT RULES:\n"
    "1. Keep every quoted string ('...' or \"...\") EXACTLY as it is — do not "
    "translate, re-spell, or re-accent it.\n"
    "2. Keep proper nouns, named entities, place names, person names, product/set "
    "names, codes, acronyms (e.g. K-12, SAT, FRPM, TOEFL), column-name mentions, "
    "and all numbers and dates EXACTLY as they appear.\n"
    "3. Do not add, remove, or explain anything.\n"
    "4. Return ONLY the Spanish translation, nothing else."
)


def derive_sample(bird_path: Path, num_queries: int, seed: int):
    """Return the exact sampled query dicts, via the SAME code the eval uses."""
    from evaluation.bird_loader import BIRDLoader
    from evaluation.sampling import sample_queries

    queries = BIRDLoader(bird_path).load_queries()
    if num_queries >= len(queries):
        return sorted(queries, key=lambda q: q["question_id"])
    return sample_queries(queries, num_queries, seed=seed, stratify=True)


def make_openai_translator(model: str):
    """Return a translate(text) -> str callable backed by the OpenAI API."""
    from openai import OpenAI  # lazy: tests inject a fake translator instead

    client = OpenAI()

    def translate(text: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": TRANSLATOR_SYSTEM},
                {"role": "user", "content": text},
            ],
        )
        return resp.choices[0].message.content.strip()

    return translate


def _quoted_strings(text: str):
    return re.findall(r"[\"']([^\"']{2,})[\"']", text or "")


def load_checkpoint(path: Path):
    """{question_id: es} from a translations.jsonl checkpoint (missing = {})."""
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["question_id"]] = rec["es"]
    return done


def translate_sample(sample, translate_fn, ckpt_path: Path):
    """Translate every sampled question, checkpointing as we go.

    Returns {question_id: spanish_question}. Already-checkpointed ids are
    skipped, so an interrupted run resumes where it stopped.
    """
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_checkpoint(ckpt_path)
    todo = [q for q in sample if q["question_id"] not in done]
    logger.info(f"Translating {len(todo)} questions ({len(done)} already done)")

    with open(ckpt_path, "a", encoding="utf-8") as ckpt:
        for i, q in enumerate(todo, 1):
            en = q["question"]
            es = translate_fn(en)
            # Light fidelity check: quoted literals must survive verbatim.
            for lit in _quoted_strings(en):
                if lit not in es:
                    logger.warning(
                        f"q{q['question_id']}: quoted literal '{lit}' not "
                        f"preserved verbatim in translation — review this row"
                    )
            done[q["question_id"]] = es
            ckpt.write(json.dumps(
                {"question_id": q["question_id"], "en": en, "es": es},
                ensure_ascii=False) + "\n")
            ckpt.flush()
            if i % 10 == 0 or i == len(todo):
                logger.info(f"  {i}/{len(todo)} translated")
    return done


def write_mirror(bird_path: Path, out_path: Path, translations: dict) -> int:
    """Write <out>/dev.json = full dev.json copy with sampled questions replaced.

    Record ids, ORDER, evidence, SQL, and difficulty are byte-identical to the
    source — this is what guarantees the sampler re-selects the identical 100.
    Returns the number of replaced questions.
    """
    records = json.loads((bird_path / "dev.json").read_text(encoding="utf-8"))
    replaced = 0
    for rec in records:
        es = translations.get(rec.get("question_id"))
        if es:
            rec["question"] = es
            replaced += 1
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "dev.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return replaced


def link_companions(bird_path: Path, out_path: Path) -> None:
    """Symlink dev_databases/ and dev_tables.json into the mirror dir."""
    for name in ("dev_databases", "dev_tables.json"):
        src, dst = bird_path / name, out_path / name
        if dst.exists() or dst.is_symlink():
            continue
        if not src.exists():
            logger.warning(f"{src} not found — link skipped (create it manually)")
            continue
        try:
            dst.symlink_to(os.path.relpath(src, out_path))
            logger.info(f"linked {dst} -> {src}")
        except OSError as e:  # e.g. filesystem without symlink support
            logger.warning(f"symlink failed ({e}); copy manually: cp -r {src} {dst}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bird-path", default="data/bird", type=Path)
    ap.add_argument("--out-path", default="data/bird_es", type=Path)
    ap.add_argument("--num-queries", default=100, type=int)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--model", default="gpt-4.1-mini",
                    help="OpenAI model used for translation (temperature 0)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Derive + print the sample ids and 3 example questions; write nothing")
    args = ap.parse_args()

    sample = derive_sample(args.bird_path, args.num_queries, args.seed)
    ids = [q["question_id"] for q in sample]
    logger.info(f"Derived {len(ids)} sampled question_ids (seed={args.seed}): {ids}")

    if args.dry_run:
        for q in sample[:3]:
            logger.info(f"  q{q['question_id']} [{q.get('difficulty')}]: {q['question']}")
        return 0

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY is not set — translation needs the API")
        return 1

    translate = make_openai_translator(args.model)
    translations = translate_sample(sample, translate, args.out_path / "translations.jsonl")
    replaced = write_mirror(args.bird_path, args.out_path, translations)
    link_companions(args.bird_path, args.out_path)

    logger.info(f"✓ {args.out_path}/dev.json written ({replaced} questions in Spanish)")
    logger.info(
        "Run the ablation:\n  python run_full_evaluation.py --config config.yaml "
        f"--seed {args.seed} --num_queries {args.num_queries} --stratify "
        f"--bird_path {args.out_path} --output_name local_{args.num_queries}_es"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
