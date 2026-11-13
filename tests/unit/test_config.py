import pytest

from ragkit.config import deep_merge, llm_settings, load_config, parse_overrides


@pytest.fixture
def configs(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "name: base\n"
        "chunker: {impl: markdown_heading, max_tokens: 400}\n"
        "embedder: {impl: hashing, dim: 64}\n"
        "reranker: {impl: none}\n"
    )
    (tmp_path / "child.yaml").write_text(
        "extends: base.yaml\nname: child\nreranker: {impl: lexical}\nabstention: {enabled: false}\n"
    )
    (tmp_path / "rechunk.yaml").write_text("extends: child.yaml\nchunker: {impl: fixed}\n")
    return tmp_path


def test_extends_deep_merges_and_child_wins(configs):
    cfg = load_config(configs / "child.yaml")
    assert cfg.name == "child"
    assert cfg.reranker.impl == "lexical"
    assert cfg.chunker.params == {"max_tokens": 400}
    assert cfg.abstention.enabled is False


def test_index_is_shared_until_chunking_or_embedding_changes(configs):
    base, child = load_config(configs / "base.yaml"), load_config(configs / "child.yaml")
    assert base.index_key() == child.index_key()
    assert load_config(configs / "rechunk.yaml").index_key() != child.index_key()


def test_cli_overrides_are_parsed_as_yaml(configs):
    over = parse_overrides(["retrieval.context_size=8", "abstention.enabled=false"])
    assert over == {"retrieval": {"context_size": 8}, "abstention": {"enabled": False}}
    cfg = load_config(configs / "base.yaml", over)
    assert cfg.retrieval.context_size == 8


def test_circular_extends_is_an_error(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path / "a.yaml")


def test_judge_settings_fall_back_to_generator_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_MODEL", "cheap-model")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_STRUCTURED_MODES", "json_object,text")
    monkeypatch.setenv("JUDGE_MODEL", "judge-model")
    gen, judge = llm_settings("generator"), llm_settings("judge")
    assert gen.model == "cheap-model" and gen.structured_modes == ["json_object", "text"]
    assert judge.model == "judge-model" and judge.api_key == "k"


def test_empty_env_values_mean_unset(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("JUDGE_PROVIDER", "")
    monkeypatch.setenv("JUDGE_BASE_URL", "")
    judge = llm_settings("judge")
    assert judge.provider == "openai_compatible" and judge.model == "m"


def test_changing_impl_replaces_the_stage_instead_of_merging_params():
    base = {"chunker": {"impl": "markdown_heading", "min_tokens": 80}, "x": {"a": 1}}
    out = deep_merge(base, {"chunker": {"impl": "fixed", "max_tokens": 300}, "x": {"b": 2}})
    assert out["chunker"] == {"impl": "fixed", "max_tokens": 300}
    assert out["x"] == {"a": 1, "b": 2}
    same = deep_merge(base, {"chunker": {"impl": "markdown_heading", "max_tokens": 300}})
    assert same["chunker"] == {"impl": "markdown_heading", "min_tokens": 80, "max_tokens": 300}


def test_plugins_are_imported_so_project_stages_register(tmp_path, monkeypatch):
    (tmp_path / "my_stages.py").write_text(
        "from pydantic import BaseModel\n"
        "from ragkit.core.registry import RERANKERS\n\n"
        "class P(BaseModel):\n    pass\n\n"
        "@RERANKERS.register('reverse', P)\n"
        "def _reverse(params, **_):\n    return 'reverse-reranker'\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    from ragkit.config import RAGConfig
    from ragkit.core.registry import RERANKERS

    cfg = RAGConfig.model_validate({"plugins": ["my_stages"], "reranker": {"impl": "reverse"}})
    assert RERANKERS.create(cfg.reranker) == "reverse-reranker"


def test_unknown_plugin_is_a_config_error():
    import pydantic
    import pytest

    from ragkit.config import RAGConfig

    with pytest.raises(pydantic.ValidationError, match="no_such_module"):
        RAGConfig.model_validate({"plugins": ["no_such_module"]})


def test_a_typo_in_a_config_key_is_an_error():
    import pydantic
    import pytest

    from ragkit.config import RAGConfig

    with pytest.raises(pydantic.ValidationError, match="min_scor"):
        RAGConfig.model_validate({"abstention": {"min_scor": 0.5}})
    with pytest.raises(pydantic.ValidationError, match="retriver"):
        RAGConfig.model_validate({"retriver": {"impl": "bm25"}})


def test_judge_can_set_a_zero_price_explicitly(monkeypatch):
    monkeypatch.setenv("LLM_INPUT_PRICE_PER_MTOK", "3")
    monkeypatch.setenv("JUDGE_INPUT_PRICE_PER_MTOK", "0")
    assert llm_settings("judge").input_price_per_mtok == 0.0


def test_provider_can_name_a_plugin_client(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "my_provider")
    assert llm_settings().provider == "my_provider"
