# app/tests/test_llm.py
from app import llm


def test_build_user_message_prepends_fewshot():
    msg = llm.build_user_message("EX1\nEX2", "warfarin interactions")
    assert msg.endswith("User question: warfarin interactions")
    assert msg.startswith("EX1")
    assert "---" in msg


def test_build_prompt_is_chatml():
    p = llm.build_prompt("SYS", "USERMSG")
    assert "<|im_start|>system\nSYS<|im_end|>" in p
    assert "<|im_start|>user\nUSERMSG<|im_end|>" in p
    assert p.rstrip().endswith("<|im_start|>assistant")


def test_generate_posts_grammar_and_returns_content(monkeypatch):
    captured = {}

    def fake_post(url, payload):
        captured["url"] = url
        captured["payload"] = payload
        return {"content": 'Traverse["warfarin", "interacts_with"]'}

    monkeypatch.setattr(llm, "_post_completion", fake_post)
    out = llm.generate("http://127.0.0.1:8081", "SYS", "USERMSG",
                       grammar="root ::= \"x\"", max_tokens=128, temperature=0.0)
    assert out == 'Traverse["warfarin", "interacts_with"]'
    assert captured["payload"]["grammar"] == 'root ::= "x"'
    assert captured["payload"]["n_predict"] == 128
    assert captured["payload"]["temperature"] == 0.0
    assert captured["url"].endswith("/completion")
