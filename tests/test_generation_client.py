from rag_turkish_law.generation import client


def test_ollama_chat_uses_native_api(monkeypatch):
    calls = []

    def fake_retry(messages, model_name, *, max_new_tokens, temperature, retries):
        calls.append((messages, model_name, max_new_tokens, temperature, retries))
        return "tamam"

    def fail_get_client(_model=None):
        raise AssertionError("ollama provider should not use InferenceClient")

    monkeypatch.setattr(client, "_retry_ollama_chat", fake_retry)
    monkeypatch.setattr(client, "get_client", fail_get_client)

    text = client.chat(
        [{"role": "user", "content": "Merhaba"}],
        model="qwen3.6:27b",
        max_new_tokens=32,
        temperature=0.1,
        retries=1,
    )

    assert text == "tamam"
    assert calls == [
        ([{"role": "user", "content": "Merhaba"}], "qwen3.6:27b", 32, 0.1, 1)
    ]
