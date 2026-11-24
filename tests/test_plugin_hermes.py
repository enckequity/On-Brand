"""Load On-Brand through Hermes' real PluginManager and drive its hooks, tools and command.

Skipped when Hermes is not importable. Run against a checkout:
    HERMES_AGENT_REPO=../hermes-agent PYTHONPATH=../hermes-agent python -m pytest -c tests/pytest.ini --rootdir tests tests
"""

import json
import shutil
from pathlib import Path

import pytest

plugins = pytest.importorskip("hermes_cli.plugins")

ROOT = Path(__file__).resolve().parents[1]
GOOD = ("Thanks for reaching out to Summit Plumbing & Heating. A licensed tech can come by Thursday "
        "between 8 and 10 am.")


@pytest.fixture
def loaded(tmp_path, monkeypatch):
    home = tmp_path / "home"
    dest = home / "plugins" / "on-brand"
    shutil.copytree(ROOT, dest, ignore=shutil.ignore_patterns(".git", "tests", "__pycache__"))
    (home / "config.yaml").write_text("plugins:\n  enabled: [on-brand]\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = plugins.PluginManager()
    manager.discover_and_load()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    loaded = manager._plugins["on-brand"]
    module = loaded.module
    calls = []

    def fake_llm(messages, timeout):
        calls.append(messages)
        return GOOD

    monkeypatch.setattr(module, "_llm", fake_llm)
    return manager, module, calls


def test_registers_everything(loaded):
    manager, module, _ = loaded
    info = manager._plugins["on-brand"]
    assert sorted(info.tools_registered) == ["onbrand_check", "onbrand_learn", "onbrand_rules",
                                             "onbrand_voiceover"]
    assert set(info.hooks_registered) >= {"pre_llm_call", "transform_llm_output", "pre_tool_call"}


def test_full_flow(loaded):
    manager, module, calls = loaded
    assert "off" in module.handle_command("")
    assert "Summit Plumbing" in module.handle_command("starter")
    assert "Glow Med Spa" in module.handle_command("list")
    assert "ON for Summit Plumbing & Heating" in module.handle_command("use summit-plumbing")

    ctx = plugins.invoke_hook("pre_llm_call", session_id="s", user_message="write a tweet")
    assert any(isinstance(r, dict) and "```copy" in r.get("context", "") for r in ctx)

    reply = "Here's one:\n```copy\nWe're the cheapest plumbers in town!!! Call now!!!\n```"
    out = [r for r in plugins.invoke_hook("transform_llm_output", response_text=reply,
                                          session_id="s", model="m", platform="cli", turn_id="t1")
           if isinstance(r, str)]
    assert out and GOOD in out[0] and calls

    block, _ = plugins._dispatch_pre_tool_call_hooks(
        "x_post_tweet", {"text": "We're the cheapest in town!!! Best in town!!!"}, session_id="s")
    assert block and block.startswith("Not sent:") and "Hermes" not in block and "x_post_tweet" not in block
    ok, _ = plugins._dispatch_pre_tool_call_hooks("x_post_tweet", {"text": GOOD}, session_id="s")
    assert ok is None
    other, _ = plugins._dispatch_pre_tool_call_hooks("read_file", {"path": "x"}, session_id="s")
    assert other is None

    log = module.handle_command("log")
    assert "[blocked]" in log and "[fixed]" in log


def test_tools(loaded):
    manager, module, _ = loaded
    module.handle_command("starter")
    res = json.loads(module.tool_check({"text": "We're the cheapest!!!", "brand": "summit-plumbing"}))
    assert res["passed"] is False and res["score"] < 100
    res = json.loads(module.tool_check({"text": "We're the cheapest!!!", "brand": "summit-plumbing", "fix": True}))
    assert res["fixed_text"] == GOOD
    res = json.loads(module.tool_rules({"brand": "Acme Labs", "person": "third", "banned": ["synergy"],
                                        "activate": True}))
    assert res["created"] and res["active"]
    res = json.loads(module.tool_voiceover({"text": "Acme Labs raised $10,000 — see https://x.y", "ssml": False}))
    assert "ten thousand dollars" in res["script"] and "http" not in res["script"]


def test_nothing_happens_when_off(loaded):
    manager, module, _ = loaded
    module.handle_command("starter")
    module.handle_command("off")
    reply = "```copy\nWe are revolutionary\n```"
    out = [r for r in plugins.invoke_hook("transform_llm_output", response_text=reply, session_id="s",
                                          model="m", platform="cli", turn_id="t2") if isinstance(r, str)]
    assert out == []


def test_rewrite_goes_through_the_other_reply_filters_last(loaded):
    """TeamShift #4637: the engine keeps the first rewrite, so another filter (white-label naming)
    must see On-Brand's rewrite, not only the original reply."""
    manager, module, calls = loaded
    seen = []

    def white_label(response_text="", **_):
        seen.append(response_text)
        return response_text.replace("licensed tech", "licensed technician") if "licensed tech" in response_text else None

    manager._hooks.setdefault("transform_llm_output", []).append(white_label)
    module.handle_command("starter")
    module.handle_command("use summit-plumbing")
    reply = "```copy\nWe're the cheapest plumbers in town!!!\n```"
    out = [r for r in plugins.invoke_hook("transform_llm_output", response_text=reply, session_id="s",
                                          model="m", platform="cli", turn_id="t3") if isinstance(r, str)]
    assert out and "licensed technician" in out[0] and "cheapest" not in out[0]
    assert any(GOOD in s for s in seen)

    def broken(response_text="", **_):
        raise RuntimeError("filter down")

    manager._hooks["transform_llm_output"] = [module._on_transform_llm_output, broken]
    out = [r for r in plugins.invoke_hook("transform_llm_output", response_text=reply, session_id="s",
                                          model="m", platform="cli", turn_id="t4") if isinstance(r, str)]
    assert out == []  # the rewrite is not delivered without the filter having seen it


def test_new_brand_hint_does_not_name_the_host(loaded):
    _, module, _ = loaded
    out = module.handle_command("new Acme Roofing")
    assert "Hermes" not in out and "ask in chat" in out
