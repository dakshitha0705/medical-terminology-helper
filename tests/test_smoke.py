"""These pass on day one. They check the plumbing so you notice quickly if you break it."""
from app import core, llm
from app.storage import Store


def test_the_store_keeps_what_you_put_in_it(tmp_path):
    store = Store(str(tmp_path / "app.json"))
    store.put("hello", {"n": 1})
    assert store.get("hello") == {"n": 1}
    assert store.get("missing", "fallback") == "fallback"
    assert store.all() == {"hello": {"n": 1}}


def test_the_model_is_local_and_needs_no_key(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "none.pt"))
    assert llm.model_path() == tmp_path / "none.pt"
    try:
        llm.MedicalGPT()
        raise AssertionError("expected ModelNotFoundError")
    except core.ModelNotFoundError as e:
        assert "scripts/train.py" in str(e)


def test_the_command_line_runs(capsys):
    from app.cli import main
    main(["hello"])
    assert "is ready" in capsys.readouterr().out
