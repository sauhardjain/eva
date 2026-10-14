"""Tests for the Cartesia managed server's session_create, transcript pieces and audio mix."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from eva.assistant.cartesia_managed_server import CartesiaManagedAssistantServer, _mix_pcm16, _SpeechSegment
from eva.utils.audio_utils import pcm16_mix


def test_launcher_excludes_unused_component_settings(tmp_path):
    """Saved run configs must not inherit unrelated component models from .env."""
    script = Path(__file__).resolve().parents[3] / "scripts/cartesia_managed/run.sh"
    (tmp_path / "output").mkdir()
    (tmp_path / "output/cartesia_managed_agents.json").write_text(
        json.dumps({"model": "test-reasoner", "agents": {"airline": "agent_test"}})
    )
    (tmp_path / ".env").write_text(
        "CARTESIA_API_KEY=test-key\n"
        "EVA_MODEL__STT=unused-recognizer\n"
        'EVA_MODEL__STT_PARAMS=\'{"model":"unused-recognizer","api_key":"unused-key"}\'\n'
        "EVA_MODEL__TTS=unused-speaker\n"
        'EVA_MODEL__TTS_PARAMS=\'{"model":"unused-speaker","api_key":"unused-key"}\'\n'
        "EVA_MODEL__AUDIO_LLM=unused-audio-model\n"
        'EVA_MODEL__AUDIO_LLM_PARAMS=\'{"model":"unused-audio-model","api_key":"unused-key"}\'\n'
    )
    # Exercise the real launcher and settings loader without starting a paid benchmark.
    uv = tmp_path / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if sys.argv[1:3] == ['run', 'python']:\n"
        "    os.execv(sys.executable, [sys.executable, *sys.argv[3:]])\n"
        "assert sys.argv[1:3] == ['run', 'eva']\n"
        "from eva.models.config import RunConfig\n"
        "config = RunConfig(_env_file='.env')\n"
        "open('captured.json', 'w').write(json.dumps(config.model.model_dump()))\n"
    )
    uv.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("EVA_", "CARTESIA_"))}
    env["PATH"] = f"{tmp_path}{os.pathsep}{env['PATH']}"
    subprocess.run(["bash", str(script), "airline"], cwd=tmp_path, env=env, check=True, capture_output=True)
    model = json.loads((tmp_path / "captured.json").read_text())
    for component in ("stt", "tts", "audio_llm"):
        assert not model[component]
        assert model[f"{component}_params"] is None
    assert model["s2s"] == "cartesia_managed"
    assert model["s2s_params"]["alias"] == "test-reasoner+cartesia-managed-agents"
    assert "unused-" not in json.dumps(model)


def _bare_server() -> CartesiaManagedAssistantServer:
    srv = object.__new__(CartesiaManagedAssistantServer)
    srv._system_prompt = "you are a helpful agent"
    srv.initial_message = "Hello, how can I help you today?"
    return srv


class TestSessionCreate:
    def test_streams_continuous_16k_pcm(self):
        event = _bare_server()._session_create()
        assert event["type"] == "session_create"
        assert event["audio"] == {"input_format": "pcm_16000", "continuous_input": True}

    def test_sends_eva_prompt_and_greeting_as_dynamic_variables(self):
        event = _bare_server()._session_create()
        assert event["dynamic_variables"] == {
            "system_prompt": "you are a helpful agent",
            "initial_message": "Hello, how can I help you today?",
        }


class TestSpeechSegment:
    def test_new_agent_message_starts_a_piece_stamped_when_spoken(self, monkeypatch):
        segment = _SpeechSegment(1000)
        segment.add_text("Submitting")
        segment.add_text(" that now.")
        assert segment.take_pieces() == []
        monkeypatch.setattr("eva.assistant.cartesia_managed_server._now_ms", lambda: 2500)
        segment.add_text("Your request is in.")
        segment.add_text(" The number is R E Q one.")
        assert segment.take_pieces() == [("Submitting that now.", 1000)]
        segment.end_piece()
        assert segment.take_pieces() == [("Your request is in. The number is R E Q one.", 2500)]
        assert segment.text == "Submitting that now. Your request is in. The number is R E Q one."

    def test_tool_call_closes_the_piece_and_restamps_the_next(self, monkeypatch):
        segment = _SpeechSegment(1000)
        segment.add_text("One moment while I check")
        segment.end_piece()
        monkeypatch.setattr("eva.assistant.cartesia_managed_server._now_ms", lambda: 4000)
        segment.add_text(" and you are all set")
        segment.end_piece()
        assert segment.take_pieces() == [("One moment while I check", 1000), ("and you are all set", 4000)]

    def test_blank_text_makes_no_piece(self):
        segment = _SpeechSegment(1000)
        segment.add_text("  ")
        segment.end_piece()
        assert segment.take_pieces() == []


class TestMixPcm16:
    def test_matches_eva_mix_with_clipping_and_unequal_lengths(self):
        rng = np.random.default_rng(0)
        a = rng.integers(-32768, 32767, 1000, dtype=np.int16).tobytes()
        b = rng.integers(-32768, 32767, 700, dtype=np.int16).tobytes()
        assert _mix_pcm16(a, b) == pcm16_mix(a, b)
        assert _mix_pcm16(b, a) == pcm16_mix(b, a)
