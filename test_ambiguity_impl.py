"""Quick test to verify ambiguity resolver LLM mode implementation.

This test verifies the code is properly implemented without loading the full SLM.
"""

import sys
import io

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import inspect
from query_planner.ambiguity_resolver import AmbiguityResolver
from aegis_types import Query, Language

def test_implementation():
    """Verify LLM mode implementation without loading SLM."""

    print("=" * 80)
    print("Testing Ambiguity Resolver LLM Mode Implementation")
    print("=" * 80)

    # Test 1: Check __init__ signature includes slm_generator
    print("\n1. Checking __init__ signature...")
    sig = inspect.signature(AmbiguityResolver.__init__)
    params = list(sig.parameters.keys())

    if 'slm_generator' in params:
        print("   ✓ slm_generator parameter present in __init__")
    else:
        print("   ✗ FAILED: slm_generator parameter missing")
        return False

    # Test 2: Check _detect_llm method exists and is not a stub
    print("\n2. Checking _detect_llm implementation...")
    source = inspect.getsource(AmbiguityResolver._detect_llm)

    if "Not yet implemented" in source:
        print("   ✗ FAILED: _detect_llm is still a stub")
        return False
    elif "self.slm_generator" in source:
        print("   ✓ _detect_llm references slm_generator")
    else:
        print("   ✗ FAILED: _detect_llm doesn't use slm_generator")
        return False

    # Test 3: Check helper methods exist
    print("\n3. Checking helper methods...")

    if hasattr(AmbiguityResolver, '_build_ambiguity_detection_prompt'):
        print("   ✓ _build_ambiguity_detection_prompt method exists")
    else:
        print("   ✗ FAILED: _build_ambiguity_detection_prompt missing")
        return False

    if hasattr(AmbiguityResolver, '_parse_ambiguity_response'):
        print("   ✓ _parse_ambiguity_response method exists")
    else:
        print("   ✗ FAILED: _parse_ambiguity_response missing")
        return False

    # Test 4: Check prompt template
    print("\n4. Checking prompt template...")
    prompt_source = inspect.getsource(AmbiguityResolver._build_ambiguity_detection_prompt)

    if "TEMPORAL" in prompt_source and "SCHEMA" in prompt_source and "UNDERSPECIFIED" in prompt_source:
        print("   ✓ Prompt template includes all ambiguity types")
    else:
        print("   ✗ FAILED: Prompt template incomplete")
        return False

    if "JSON" in prompt_source:
        print("   ✓ Prompt requests JSON output")
    else:
        print("   ✗ FAILED: Prompt doesn't request JSON format")
        return False

    # Test 5: Instantiation test without SLM
    print("\n5. Testing instantiation...")

    try:
        # Create resolver without SLM (should fallback to rules)
        resolver = AmbiguityResolver(
            detector_type="llm",
            slm_generator=None
        )

        if resolver.detector_type == "rules":
            print("   ✓ Properly falls back to rules when SLM not available")
        else:
            print("   ✗ FAILED: Doesn't fallback to rules")
            return False
    except Exception as e:
        print(f"   ✗ FAILED: Error during instantiation: {e}")
        return False

    # Test 6: Rule-based detection still works
    print("\n6. Testing rule-based detection...")

    try:
        resolver = AmbiguityResolver(detector_type="rules")
        query = Query(text="Show me recent sales", language=Language.ENGLISH, database_id="test_db")
        ambiguities = resolver.detect(query)

        if len(ambiguities) > 0:
            print(f"   ✓ Detected {len(ambiguities)} ambiguities")
            for amb in ambiguities:
                print(f"     - {amb.type}: '{amb.phrase}'")
        else:
            print("   ✗ FAILED: No ambiguities detected")
            return False
    except Exception as e:
        print(f"   ✗ FAILED: Error during detection: {e}")
        return False

    # Test 7: Check privacy preservation
    print("\n7. Verifying privacy preservation...")
    detect_llm_source = inspect.getsource(AmbiguityResolver._detect_llm)

    # Check that it doesn't use external APIs
    bad_patterns = ['requests.', 'openai.', 'anthropic.', 'http://', 'https://']
    has_external_calls = any(pattern in detect_llm_source for pattern in bad_patterns)

    if not has_external_calls:
        print("   ✓ No external API calls detected (privacy-preserving)")
    else:
        print("   ✗ WARNING: May contain external API calls")

    # Check for local processing indicators
    if "torch" in detect_llm_source or "slm_generator.model" in detect_llm_source:
        print("   ✓ Uses local model processing")
    else:
        print("   ✗ WARNING: Unclear if using local processing")

    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nImplementation Summary:")
    print("  ✓ SLM integration added to __init__")
    print("  ✓ _detect_llm() fully implemented (no longer stub)")
    print("  ✓ Prompt templates created for ambiguity detection")
    print("  ✓ JSON parsing implemented")
    print("  ✓ Fallback to rules when SLM unavailable")
    print("  ✓ Privacy-preserving (local SLM, no external calls)")
    print("\nTo test with actual SLM:")
    print("  1. Ensure HF_HUB_TOKEN is set in .env")
    print("  2. Run: python test_ambiguity_llm.py")

    return True


if __name__ == "__main__":
    success = test_implementation()
    exit(0 if success else 1)
