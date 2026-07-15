"""Trained pairwise selection judge (the CHASE-SQL selector lever).

Wraps the fine-tuned `aegis-sql-selector-3b` model (Qwen2.5-Coder-3B + LoRA,
merged) trained by the `aegis-selector` notebook. Given the question, a reduced
schema, and two candidate SQLs *with their execution-result previews*, the model
replies a single token — `A` or `B` — for the candidate that answers correctly.

We turn that binary judge into a winner over a pool via a **round-robin
tournament with both orderings** (each unordered pair is compared A-vs-B and
B-vs-A) to cancel position bias; the candidate with the most wins is selected,
ties broken toward the execution-vote / greedy candidate the caller passes first.

This module is intentionally self-contained inside ``agents/`` — it is the
booster's optional accuracy lever, so it is removed together with the booster if
the multi-agent pipeline is retired. It has no import-time torch dependency (the
model is loaded lazily on first use), so the rest of the package still imports
offline.

Prompt format is byte-compatible with the notebook's training prompt
(`SELECTOR_SYSTEM` + `selector_prompt`); changing it here silently degrades the
trained model, so keep the two in sync.
"""

import re
import sqlite3
import time
from typing import Dict, List, Optional, Tuple

try:  # pragma: no cover - loguru is always present in this repo
    from loguru import logger
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("aegis.pairwise_selector")


# --- prompt (MUST match aegis-selector/selector_finetune.ipynb) ----------------

SELECTOR_SYSTEM = (
    "You are an expert SQL judge. Given a database schema, a question, and two "
    "candidate SQLite queries with their execution results, decide which candidate "
    "answers the question correctly. Reply with exactly one letter: A or B."
)


def build_selector_prompt(
    question: str, evidence: str, schema_block: str,
    sql_a: str, prev_a: str, sql_b: str, prev_b: str,
) -> str:
    """The pairwise comparison prompt (schema already reduced to relevant tables)."""
    q = f"{evidence.strip()}\n{question.strip()}" if (evidence or "").strip() else question.strip()
    return (
        f"Database schema:\n{schema_block}\n\nQuestion: {q}\n\n"
        f"Candidate A:\n{sql_a}\nExecution result A: {prev_a}\n\n"
        f"Candidate B:\n{sql_b}\nExecution result B: {prev_b}\n\n"
        f"Which candidate answers the question correctly? Reply with exactly A or B."
    )


# --- execution previews --------------------------------------------------------

def execute_preview(db_path: str, sql: str, timeout: float = 15.0,
                    max_rows: int = 20, max_chars: int = 1000) -> str:
    """Run a candidate and return a compact, truncated result preview string.

    Mirrors the notebook's execution-preview normalization: a bounded number of
    rows, stringified, clipped to ``max_chars``. Errors become ``"ERROR: ..."``
    so the judge can penalize a non-executing candidate.
    """
    if not sql or not sql.strip():
        return "ERROR: empty query"
    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        deadline = time.time() + timeout
        conn.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 1000)
        cur = conn.execute(sql)
        rows = cur.fetchmany(max_rows)
        preview = str(rows)
        if len(preview) > max_chars:
            preview = preview[:max_chars] + " …(truncated)"
        return preview if rows else "(0 rows)"
    except Exception as e:  # sqlite errors, timeouts
        return f"ERROR: {str(e)[:200]}"
    finally:
        if conn is not None:
            try:
                conn.set_progress_handler(None, 0)
                conn.close()
            except Exception:
                pass


# --- shared tournament logic (backend-agnostic) --------------------------------

def parse_ab(text: str, default: str = "A") -> str:
    """Extract the model's A/B choice; default on none.

    Prefers a STANDALONE letter (word boundary) so words like "answer" don't
    register as an 'A'; takes the last such token since models often restate the
    options before committing. Falls back to any A/B, then the default.
    """
    if not text:
        return default
    upper = text.upper()
    standalone = re.findall(r"\b[AB]\b", upper)
    if standalone:
        return standalone[-1]
    m = re.search(r"[AB]", upper)
    return m.group(0) if m else default


def run_tournament(pool: List[str], previews: Dict[str, str], compare) -> Tuple[str, Dict[str, int]]:
    """Round-robin, both-orderings tournament. ``compare(a,pa,b,pb) -> 'A'|'B'``.

    Returns (winner_sql, wins). Tie-break favors the earliest candidate (the
    execution-vote / greedy default the caller places first).
    """
    wins = {c: 0 for c in pool}
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            a, b = pool[i], pool[j]
            wins[a if compare(a, previews[a], b, previews[b]) == "A" else b] += 1
            wins[b if compare(b, previews[b], a, previews[a]) == "A" else a] += 1
    best = max(range(len(pool)), key=lambda k: (wins[pool[k]], -k))
    return pool[best], wins


def tournament_select(candidates: List[str], *, question: str, evidence: str,
                      schema_block: str, db_path: str, compare,
                      max_candidates: int = 4, exec_timeout: float = 15.0) -> Optional[str]:
    """Execute previews + run the tournament over the top candidates.

    ``compare`` is the A/B backend (trained model logits OR an existing model's
    text reply). Returns the winner, or None when it cannot run (no db / <2).
    """
    pool = [c for c in candidates if c and c.strip()][:max_candidates]
    if len(pool) < 2:
        return pool[0] if pool else None
    if not db_path or db_path == ":memory:":
        return None
    previews = {c: execute_preview(db_path, c, exec_timeout) for c in pool}
    winner, wins = run_tournament(pool, previews, compare)
    logger.info(f"tournament_select: winner has {wins[winner]}/{2*(len(pool)-1)} wins")
    return winner


def judge_fn_compare(judge_fn, question: str, evidence: str, schema_block: str):
    """Adapt an existing model's text-generation callable into an A/B comparator.

    ``judge_fn(prompt, n, temperature, system_prompt) -> List[str]`` — the same
    trusted-model callable the heuristic judge uses (SLM on the local path, LLM on
    the remote path). This is the *no-training* selector: reuse the model already
    loaded, but decide via pairwise A/B instead of the listwise heuristic.
    """
    def compare(sql_a, prev_a, sql_b, prev_b):
        prompt = build_selector_prompt(question, evidence, schema_block,
                                       sql_a, prev_a, sql_b, prev_b)
        try:
            out = judge_fn(prompt, 1, 0.0, SELECTOR_SYSTEM)
            return parse_ab(out[0] if out else "")
        except Exception as e:
            logger.debug(f"judge_fn_compare failed ({e}); defaulting to A")
            return "A"
    return compare


# --- lazy model cache (one instance per model id per process) ------------------

_MODEL_CACHE: Dict[str, Tuple[object, object]] = {}


def _load_model(model_id: str, cache_dir: Optional[str], hf_token: Optional[str],
                device: str, torch_dtype: str):
    """Lazily load (tokenizer, model); cached per model id. Imports torch here."""
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}.get(torch_dtype, torch.float16)
    logger.info(f"PairwiseSelector: loading trained selector '{model_id}' ({torch_dtype})")
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache_dir, token=hf_token,
        torch_dtype=dtype, device_map=device,
    )
    model.eval()
    _MODEL_CACHE[model_id] = (tok, model)
    return tok, model


class PairwiseSelector:
    """Round-robin, both-orderings tournament over a candidate pool."""

    def __init__(self, model_id: str, *, cache_dir: Optional[str] = None,
                 hf_token: Optional[str] = None, device: str = "auto",
                 torch_dtype: str = "float16", max_candidates: int = 4,
                 exec_timeout: float = 15.0) -> None:
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.hf_token = hf_token
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_candidates = max_candidates
        self.exec_timeout = exec_timeout
        self._tok = None
        self._model = None
        # cache the token ids for 'A'/'B' after the tokenizer loads
        self._a_ids = None
        self._b_ids = None

    def _ensure(self) -> bool:
        if self._model is not None:
            return True
        try:
            self._tok, self._model = _load_model(
                self.model_id, self.cache_dir, self.hf_token, self.device, self.torch_dtype
            )
            self._a_ids = self._letter_ids("A")
            self._b_ids = self._letter_ids("B")
            return True
        except Exception as e:  # missing model, OOM, offline
            logger.warning(f"PairwiseSelector disabled — could not load '{self.model_id}': {e}")
            return False

    def _letter_ids(self, letter: str) -> set:
        """Token ids that decode to the target letter (with/without leading space)."""
        ids = set()
        for variant in (letter, " " + letter):
            enc = self._tok.encode(variant, add_special_tokens=False)
            if enc:
                ids.add(enc[-1])
        return ids

    def select(self, candidates: List[str], *, question: str, evidence: str,
               schema_block: str, db_path: str) -> Optional[str]:
        """Return the tournament-winning candidate SQL, or None if unavailable.

        ``candidates`` should already be deduplicated (distinct result groups);
        the first is treated as the execution-vote/greedy default for tie-breaks.
        """
        pool = [c for c in candidates if c and c.strip()][: self.max_candidates]
        if len(pool) < 2:
            return pool[0] if pool else None
        if not db_path or db_path == ":memory:":
            return None
        if not self._ensure():
            return None

        compare = lambda a, pa, b, pb: self._compare(  # noqa: E731
            question, evidence, schema_block, a, pa, b, pb)
        return tournament_select(
            pool, question=question, evidence=evidence, schema_block=schema_block,
            db_path=db_path, compare=compare, max_candidates=self.max_candidates,
            exec_timeout=self.exec_timeout,
        )

    def _compare(self, question, evidence, schema_block, sql_a, prev_a, sql_b, prev_b) -> str:
        """One A/B comparison; returns 'A' or 'B' (defaults to 'A' on any failure)."""
        import torch

        prompt = build_selector_prompt(question, evidence, schema_block,
                                       sql_a, prev_a, sql_b, prev_b)
        try:
            messages = [
                {"role": "system", "content": SELECTOR_SYSTEM},
                {"role": "user", "content": prompt},
            ]
            text = self._tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tok(text, return_tensors="pt", truncation=True, max_length=8192)
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self._model(**inputs).logits[0, -1, :]
            a_score = max((logits[i].item() for i in self._a_ids), default=float("-inf"))
            b_score = max((logits[i].item() for i in self._b_ids), default=float("-inf"))
            return "A" if a_score >= b_score else "B"
        except Exception as e:
            logger.debug(f"PairwiseSelector comparison failed ({e}); defaulting to A")
            return "A"
