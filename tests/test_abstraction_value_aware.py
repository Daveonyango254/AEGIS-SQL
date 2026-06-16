"""Unit tests for value-aware DP abstraction.

Value-aware mode (PrivacyConfig.value_aware_abstraction, default ON) must
abstract only value-like tokens — proper nouns, quoted/numeric literals — and
never generic schema-vocabulary words such as 'name'/'price'. Legacy keyword
gating remains available when the flag is off.

Run with: python -m pytest tests/test_abstraction_value_aware.py -v
"""

from config import PrivacyConfig
from aegis_types import Language, Query
from abstraction.dp_abstractor import DPAbstractor
from abstraction.placeholder_vocab import PlaceholderVocabulary
from abstraction.reconstruction import ReconstructionModule
from abstraction.sensitivity_policy import SensitivityPolicy


# A BIRD-style question mixing schema-vocabulary words ('name', 'price') with
# genuine proper-noun values ('Adams', 'California').
QUESTION = "What is the name of the school named Adams in California with the highest price?"


def _make_abstractor(value_aware: bool) -> DPAbstractor:
    config = PrivacyConfig(value_aware_abstraction=value_aware)
    vocab = PlaceholderVocabulary(vocab_size=config.placeholder_vocab_size)
    policy = SensitivityPolicy(config.sensitivity_policy)
    return DPAbstractor(config, vocab, policy, embedding_model=None)


def _abstract(value_aware: bool):
    abstractor = _make_abstractor(value_aware)
    query = Query(text=QUESTION, language=Language.ENGLISH, database_id="california_schools")
    return abstractor.abstract(query, schema_elements=[])


def test_value_aware_skips_schema_vocabulary_words():
    """Generic schema words 'name'/'price' must survive abstraction verbatim."""
    abstracted, _ = _abstract(value_aware=True)
    originals = set(abstracted.placeholder_map.values())
    assert "name" not in originals
    assert "price" not in originals
    # And they remain readable in the text the remote LLM would see.
    assert "name" in abstracted.text
    assert "price" in abstracted.text


def test_value_aware_abstracts_proper_noun_values():
    """Proper-noun values 'Adams'/'California' must be abstracted."""
    abstracted, recon_map = _abstract(value_aware=True)
    originals = set(abstracted.placeholder_map.values())
    assert "Adams" in originals
    # 'California?' keeps its trailing punctuation under whitespace tokenization.
    assert any(tok.startswith("California") for tok in originals)
    assert abstracted.num_substitutions == len(recon_map.placeholder_to_real)
    assert abstracted.num_substitutions >= 2


def test_value_aware_reconstruction_round_trips():
    """Applying the reconstruction map restores the original question text."""
    abstracted, recon_map = _abstract(value_aware=True)
    restored = abstracted.text
    # Longest placeholders first, mirroring ReconstructionModule.reconstruct.
    for placeholder, real in sorted(
        recon_map.placeholder_to_real.items(), key=lambda kv: len(kv[0]), reverse=True
    ):
        restored = restored.replace(placeholder, real)
    assert restored == QUESTION


def test_legacy_mode_abstracts_keyword_schema_words():
    """With the flag off, legacy keyword gating still abstracts 'name'/'price'."""
    abstracted, _ = _abstract(value_aware=False)
    originals = set(abstracted.placeholder_map.values())
    # Whitespace tokenization keeps attached punctuation (e.g. 'price?'), so
    # match on substring: legacy keyword gating abstracts these schema words.
    assert any("name" in tok for tok in originals)
    assert any("price" in tok for tok in originals)


if __name__ == "__main__":
    test_value_aware_skips_schema_vocabulary_words()
    test_value_aware_abstracts_proper_noun_values()
    test_value_aware_reconstruction_round_trips()
    test_legacy_mode_abstracts_keyword_schema_words()
    print("All value-aware abstraction tests passed.")
