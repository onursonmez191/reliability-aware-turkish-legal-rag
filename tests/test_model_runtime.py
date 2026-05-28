from rag_turkish_law.api import models


def test_model_state_marks_installed_and_running(monkeypatch):
    monkeypatch.setattr(
        models,
        "_installed_models",
        lambda: {
            "qwen2.5:7b-instruct": {"size": 4_700_000_000},
            "gemma4:31b": {"size": 20_000_000_000},
        },
    )
    monkeypatch.setattr(
        models,
        "_running_models",
        lambda: [
            {
                "name": "gemma4:31b",
                "size": 20_000_000_000,
                "processor": "100% GPU",
                "context": 32768,
            }
        ],
    )

    state = models.get_models_state()
    by_name = {m["name"]: m for m in state["models"]}

    assert state["ollama_status"] == "online"
    assert state["running"] == ["gemma4:31b"]
    assert by_name["qwen2.5:7b-instruct"]["installed"] is True
    assert by_name["gemma4:31b"]["running"] is True
    assert by_name["gemma4:31b"]["processor"] == "100% GPU"


def test_load_model_unloads_other_configured_models(monkeypatch):
    calls = []

    monkeypatch.setattr(models, "_running_models", lambda: [{"name": "qwen3.6:27b"}])
    monkeypatch.setattr(models, "get_models_state", lambda: {"ok": True})
    monkeypatch.setattr(
        models,
        "_generate_keepalive",
        lambda model, keep_alive: calls.append((model, keep_alive)),
    )

    state = models.load_ollama_model("gemma4:31b", keep_alive="1m")

    assert state == {"ok": True}
    assert calls == [("qwen3.6:27b", 0), ("gemma4:31b", "1m")]


def test_unload_model_waits_until_not_running(monkeypatch):
    calls = []
    running_states = [
        [{"name": "gemma4:31b"}],
        [],
    ]

    monkeypatch.setattr(
        models,
        "_generate_keepalive",
        lambda model, keep_alive: calls.append((model, keep_alive)),
    )
    monkeypatch.setattr(models, "_running_models", lambda: running_states.pop(0))
    monkeypatch.setattr(models, "get_models_state", lambda: {"running": []})
    monkeypatch.setattr(models.time, "sleep", lambda _seconds: None)

    state = models.unload_ollama_model("gemma4:31b")

    assert state == {"running": []}
    assert calls == [("gemma4:31b", 0)]


def test_unload_model_reports_still_running(monkeypatch):
    monkeypatch.setattr(models, "_generate_keepalive", lambda _model, _keep_alive: None)
    monkeypatch.setattr(models, "_running_models", lambda: [{"name": "gemma4:31b"}])
    monkeypatch.setattr(models, "_unload_timeout", lambda: 0)

    try:
        models.unload_ollama_model("gemma4:31b")
    except models.OllamaRuntimeError as exc:
        assert "still reported as running" in str(exc)
    else:
        raise AssertionError("unload should report a still-running model")


def test_unknown_model_is_rejected():
    try:
        models.validate_model_name("not-in-config:1b")
    except models.ModelConfigError as exc:
        assert "not configured" in str(exc)
    else:
        raise AssertionError("unknown model should be rejected")
