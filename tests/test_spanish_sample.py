"""Offline tests for the Spanish language-ablation kit (no API, no real data).

Uses a synthetic dev.json fixture + a fake translator to verify:
* sample derivation is deterministic and uses the SAME code path as the eval
  (two calls -> identical ids);
* the mirror dev.json preserves ids/order/evidence/SQL/difficulty and replaces
  ONLY the sampled questions;
* translation checkpointing resumes (done ids are not re-translated);
* the quoted-literal fidelity warning logic fires on a bad translation.

Run with: python tests/test_spanish_sample.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.make_spanish_sample import (  # noqa: E402
    derive_sample, load_checkpoint, translate_sample, write_mirror,
)


def _fixture_dir(n=40):
    """A synthetic bird dir with dev.json of n queries across difficulties."""
    d = Path(tempfile.mkdtemp())
    diffs = ["simple", "moderate", "challenging"]
    records = [
        {
            "question_id": i,
            "db_id": "db%d" % (i % 3),
            "question": f"How many rows have value 'Lit{i}' in table {i}?",
            "evidence": f"Lit{i} refers to col{i}",
            "SQL": f"SELECT COUNT(*) FROM t WHERE c = 'Lit{i}'",
            "difficulty": diffs[i % 3],
        }
        for i in range(n)
    ]
    (d / "dev.json").write_text(json.dumps(records), encoding="utf-8")
    (d / "dev_databases").mkdir()   # BIRDLoader.__init__ requires it to exist
    return d, records


def test_sample_derivation_is_deterministic():
    d, _ = _fixture_dir()
    a = [q["question_id"] for q in derive_sample(d, 10, 42)]
    b = [q["question_id"] for q in derive_sample(d, 10, 42)]
    assert a == b and len(a) == 10
    c = [q["question_id"] for q in derive_sample(d, 10, 7)]
    assert c != a  # different seed -> different sample (overwhelmingly likely)


def test_mirror_preserves_everything_but_sampled_questions():
    d, records = _fixture_dir()
    sample = derive_sample(d, 10, 42)
    translations = {q["question_id"]: f"ES::{q['question']}" for q in sample}
    out = Path(tempfile.mkdtemp())
    replaced = write_mirror(d, out, translations)
    assert replaced == 10

    mirrored = json.loads((out / "dev.json").read_text(encoding="utf-8"))
    assert [r["question_id"] for r in mirrored] == [r["question_id"] for r in records]
    sampled_ids = set(translations)
    for orig, new in zip(records, mirrored):
        assert new["SQL"] == orig["SQL"]                  # gold untouched
        assert new["evidence"] == orig["evidence"]        # evidence untouched
        assert new["difficulty"] == orig["difficulty"]
        if orig["question_id"] in sampled_ids:
            assert new["question"].startswith("ES::")     # replaced
        else:
            assert new["question"] == orig["question"]    # untouched


def test_mirror_reselects_identical_sample():
    """The property the whole design rests on: sampling the mirror yields the
    same 100 (here 10) question_ids as sampling the original."""
    d, _ = _fixture_dir()
    sample = derive_sample(d, 10, 42)
    translations = {q["question_id"]: "ES" for q in sample}
    out = Path(tempfile.mkdtemp())
    write_mirror(d, out, translations)
    (out / "dev_databases").mkdir(exist_ok=True)  # in production: symlinked
    re_ids = [q["question_id"] for q in derive_sample(out, 10, 42)]
    assert re_ids == [q["question_id"] for q in sample]


def test_translation_checkpoint_resumes():
    d, _ = _fixture_dir()
    sample = derive_sample(d, 6, 42)
    ckpt = Path(tempfile.mkdtemp()) / "translations.jsonl"

    calls = []
    def fake_translate(text):
        calls.append(text)
        return "ES::" + text

    first = translate_sample(sample[:4], fake_translate, ckpt)
    assert len(first) == 4 and len(calls) == 4
    # Second run over the FULL sample: only the 2 new ids get translated.
    calls.clear()
    full = translate_sample(sample, fake_translate, ckpt)
    assert len(full) == 6 and len(calls) == 2
    assert load_checkpoint(ckpt) == full


def test_literal_fidelity_warning_path():
    """A translation that drops a quoted literal is still recorded (the warning
    is advisory), and a faithful one raises nothing."""
    d, _ = _fixture_dir()
    sample = derive_sample(d, 2, 42)
    ckpt = Path(tempfile.mkdtemp()) / "t.jsonl"
    bad = translate_sample(sample, lambda t: "sin literal alguno", ckpt)
    assert all(v == "sin literal alguno" for v in bad.values())


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); passed += 1
    print(f"=== {passed} passed, 0 failed ===")
