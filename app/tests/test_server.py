# app/tests/test_server.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))

from app.domains import load_domain          # noqa: E402
from app.server import create_app            # noqa: E402


def _client(coglang_reply):
    domain = load_domain(ROOT / "app" / "domains", "drug")

    def fake_llm(system, user_message, grammar):
        return coglang_reply

    app = create_app(domain, grammar="dummy-grammar", llm_call=fake_llm)
    app.config["TESTING"] = True
    return app.test_client()


def test_ask_read_executes_and_returns_result():
    c = _client('Traverse["warfarin", "interacts_with"]')
    resp = c.post("/ask", json={"q": "warfarin interactions"})
    body = resp.get_json()
    assert body["coglang"] == 'Traverse["warfarin", "interacts_with"]'
    assert body["needs_confirm"] is False
    assert "aspirin" in body["result"]


def test_ask_write_requires_confirm_and_does_not_mutate():
    c = _client('Delete["aspirin"]')
    resp = c.post("/ask", json={"q": "delete aspirin"})
    body = resp.get_json()
    assert body["is_write"] is True
    assert body["needs_confirm"] is True
    # graph still intact:
    dom = c.get("/domain").get_json()
    assert dom["nodes"] == 18


def test_execute_confirmed_write_then_reset():
    c = _client('Delete["aspirin"]')
    c.post("/execute", json={"coglang": 'Delete["aspirin"]'})
    assert c.get("/domain").get_json()["nodes"] == 17   # mutated (aspirin removed)
    c.post("/reset")
    assert c.get("/domain").get_json()["nodes"] == 18   # restored


def test_domain_metadata():
    c = _client('Traverse["warfarin", "interacts_with"]')
    body = c.get("/domain").get_json()
    assert body["name"] == "drug"
    assert body["nodes"] == 18
    assert body["edges"] == 22
