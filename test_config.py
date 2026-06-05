"""Test configuration and model initialization.

Quick test to verify:
1. Configuration loads correctly
2. Environment variables are substituted
3. SLM generator can be initialized
4. LLM fallback can be initialized
5. Workflow graph can be built
"""

from config import AEGISConfig
from generator import SLMGenerator, LLMFallback
from workflow import build_aegis_graph


def test_config_loading():
    """Test configuration loading."""
    print("\n=== Testing Configuration Loading ===")
    config = AEGISConfig.from_yaml("config.yaml")

    print(f"Config loaded successfully")
    print(f"  Language: {config.language}")
    print(f"  SLM Model: {config.slm.model}")
    print(f"  LLM Model: {config.llm.model}")
    print(f"  Embedding Model: {config.embedding.model}")
    print(f"  HF Token: {'SET' if config.slm.hf_token and not config.slm.hf_token.startswith('${') else 'NOT SET'}")
    print(f"  OpenAI API Key: {'SET' if config.llm.api_key and not config.llm.api_key.startswith('${') else 'NOT SET'}")

    return config


def test_slm_initialization(config):
    """Test SLM generator initialization."""
    print("\n=== Testing SLM Generator Initialization ===")
    try:
        generator = SLMGenerator(config.slm)
        print(f"SLM Generator initialized")
        print(f"  Model loaded: {generator.model is not None}")
        print(f"  Tokenizer loaded: {generator.tokenizer is not None}")
        if generator.model is None:
            print(f"  Note: Running in stub mode (model not downloaded yet)")
    except Exception as e:
        print(f"SLM Generator initialization failed: {e}")
        raise


def test_llm_initialization(config):
    """Test LLM fallback initialization."""
    print("\n=== Testing LLM Fallback Initialization ===")
    try:
        fallback = LLMFallback(config.llm)
        print(f"LLM Fallback initialized")
        print(f"  Provider: {fallback.provider}")
        print(f"  Model: {config.llm.model}")
        print(f"  Timeout: {config.llm.timeout}s")
        print(f"  Max Retries: {config.llm.max_retries}")
    except Exception as e:
        print(f"LLM Fallback initialization failed: {e}")
        raise


def test_workflow_graph(config):
    """Test workflow graph building."""
    print("\n=== Testing Workflow Graph Building ===")
    try:
        graph = build_aegis_graph(config)
        print(f"Workflow graph built successfully")
        print(f"  Graph type: {type(graph).__name__}")
    except Exception as e:
        print(f"Workflow graph building failed: {e}")
        raise


def main():
    """Run all tests."""
    print("=" * 60)
    print("AEGIS-SQL Configuration Test")
    print("=" * 60)

    try:
        # Test 1: Load configuration
        config = test_config_loading()

        # Test 2: Initialize SLM generator
        test_slm_initialization(config)

        # Test 3: Initialize LLM fallback
        test_llm_initialization(config)

        # Test 4: Build workflow graph
        test_workflow_graph(config)

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run 'python run_e2e_test.py' to test the full workflow")
        print("2. First run will download models (may take 5-10 minutes)")
        print("3. Check outputs in 'evaluation/output/' directory")

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"TESTS FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
