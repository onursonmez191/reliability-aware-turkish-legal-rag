from rag_turkish_law import config


def test_empty_rag_config_env_uses_default(monkeypatch):
    config.load_config.cache_clear()
    monkeypatch.setenv("RAG_CONFIG", "")

    cfg = config.load_config()

    assert cfg.api.host == "127.0.0.1"
    assert cfg.paths.processed_dir.endswith("data/processed")
    config.load_config.cache_clear()
