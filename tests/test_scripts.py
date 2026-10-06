import base64
import io

from PIL import Image

from agentkit.llm import Reply
from agentkit.testing import ScriptedLLM, tool_reply
from scripts import check_setup, smoke_llm
from scripts.make_fake_leather import make_dataset

# ------------------------------------------------------------------ smoke_llm


def nearest_colour(rgb):
    return min(smoke_llm.COLOURS, key=lambda n: sum((a - b) ** 2 for a, b in zip(smoke_llm.COLOURS[n], rgb)))


def colour_in_last_image(messages):
    """What a vision model would see: decode the newest image in the conversation."""
    for message in reversed(messages):
        if message["role"] != "user" or isinstance(message["content"], str):
            continue
        for part in reversed(message["content"]):
            if part["type"] == "image_url":
                url = part["image_url"]["url"]
                data = base64.b64decode(url.split(",", 1)[1])
                img = Image.open(io.BytesIO(data)).convert("RGB")
                return nearest_colour(img.getpixel((img.width // 2, img.height // 2)))
    return None


def seeing_model(will_call_tool):
    """Fake model that answers with the colour it can actually decode from the images it was sent."""

    def respond(messages, tools):
        saw = colour_in_last_image(messages)
        if will_call_tool and saw is None:
            return tool_reply("get_swatch", {})
        return tool_reply("submit_answer", {"colour": saw or "red"})

    return ScriptedLLM([respond, respond, respond, respond])


def test_smoke_passes_with_a_model_that_sees_images():
    llm = seeing_model(will_call_tool=True)
    results = smoke_llm.run_smoke(llm)
    assert [r.ok for r in results] == [True, True], results


def test_smoke_flags_a_blind_model():
    blind = ScriptedLLM(
        [tool_reply("submit_answer", {"colour": "red"}), tool_reply("get_swatch", {}), tool_reply("submit_answer", {"colour": "red"})]
    )
    results = smoke_llm.run_smoke(blind)
    assert results[0].ok is False and "cannot see" in results[0].detail
    assert results[1].ok is False and "cannot see" in results[1].detail


def test_smoke_flags_a_model_that_never_calls_the_tool():
    lazy = ScriptedLLM([tool_reply("submit_answer", {"colour": "green"})] * 4)
    result = smoke_llm.stage_tool_image(lazy)
    assert not result.ok and "never called the tool" in result.detail


def test_smoke_reports_provider_errors():
    from agentkit import LLMError

    result = smoke_llm.stage_vision_input(ScriptedLLM([LLMError("401 unauthorized")]))
    assert not result.ok and "401" in result.detail


# ------------------------------------------------------------------ check_setup


def by_name(checks):
    return {c.name: c for c in checks}


def test_check_data_missing_folder(tmp_path):
    checks = check_setup.check_data(tmp_path / "nope")
    assert checks[0].status == "fail" and "make_fake_leather" in checks[0].fix


def test_check_data_on_fake_dataset(tmp_path):
    out = tmp_path / "leather"
    make_dataset(out, size=64, n_train=3, n_test_good=2, per_defect=2)
    checks = by_name(check_setup.check_data(out))
    assert checks["data kind"].status == "warn" and "FAKE" in checks["data kind"].detail
    assert checks["train/good"].detail == "3 images"
    assert checks["masks"].status == "ok"
    assert "defect names" not in checks  # names match the course
    assert checks["image size"].detail.startswith("64x64")
    assert not [c for c in checks.values() if c.status == "fail"]


def test_check_data_flags_unexpected_names_and_missing_masks(tmp_path):
    out = tmp_path / "leather"
    make_dataset(out, size=64, n_train=1, n_test_good=1, per_defect=1)
    (out / "FAKE_DATA.txt").unlink()  # pretend it is real
    (out / "test" / "cut").rename(out / "test" / "slit")
    (out / "ground_truth" / "fold" / "000_mask.png").unlink()
    checks = by_name(check_setup.check_data(out))
    assert checks["data kind"].status == "ok"
    assert checks["defect names"].status == "warn" and "slit" in checks["defect names"].detail
    assert checks["masks"].status == "warn"


def test_check_llm_without_keys_is_a_warning_not_a_failure():
    check = check_setup.check_llm({"LLM_PROVIDERS": "gemini"})
    assert check.status == "warn" and "GEMINI_API_KEY" in check.detail


def test_check_llm_with_key_is_ok():
    check = check_setup.check_llm({"LLM_PROVIDERS": "gemini", "GEMINI_API_KEY": "x"})
    assert check.status == "ok"


def test_check_python_versions():
    assert check_setup.check_python((3, 10)).status == "fail"
    assert check_setup.check_python((3, 11)).status == "ok"


def test_render_summarises_and_main_exit_codes(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDERS", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert check_setup.main(["--data", str(tmp_path / "missing")]) == 1
    assert "blocking problem" in capsys.readouterr().out
    out = tmp_path / "leather"
    make_dataset(out, size=64, n_train=2, n_test_good=2, per_defect=1)
    assert check_setup.main(["--data", str(out)]) == 0


def test_reply_type_still_importable():
    assert Reply is not None


def test_download_listing_rejects_unsafe_paths():
    from scripts.download_mvtec_leather import safe_entries

    listing = [
        {"type": "file", "path": "leather/train/good/000.png", "size": 10},
        {"type": "directory", "path": "leather/train"},
        {"type": "file", "path": "leather/../evil.png", "size": 1},
        {"type": "file", "path": "/leather/abs.png", "size": 1},
        {"type": "file", "path": "wood/train/good/000.png", "size": 1},
    ]
    assert safe_entries(listing) == [("leather/train/good/000.png", 10)]


def test_download_skips_complete_files(tmp_path, monkeypatch):
    from scripts import download_mvtec_leather as d

    target = tmp_path / "leather" / "x.png"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"12345")
    monkeypatch.setattr(d.urllib.request, "urlretrieve", lambda *a: (_ for _ in ()).throw(AssertionError("no network")))
    assert d.download_one("leather/x.png", 5, tmp_path) is None
