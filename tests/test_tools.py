import json

from agentkit.tools import MAX_RESULT_CHARS, ToolError, ToolRegistry


def make_registry():
    registry = ToolRegistry()

    @registry.register
    def add(a: int, b: int = 1) -> int:
        """Add two integers."""
        return a + b

    @registry.register
    def boom() -> str:
        """Always fails."""
        return str(1 / 0)

    @registry.register
    def refuse(x: str) -> str:
        """Raises a model-facing error."""
        raise ToolError(f"cannot handle {x!r}, try 'y'")

    @registry.register
    def big() -> str:
        """Returns a huge string."""
        return "x" * (MAX_RESULT_CHARS + 500)

    @registry.register
    def lookup(name: str, limit: int | None = None) -> dict:
        """Optional parameter."""
        return {"name": name, "limit": limit}

    return registry


def test_spec_is_clean_and_complete():
    specs = {s["function"]["name"]: s["function"] for s in make_registry().specs()}
    add = specs["add"]
    assert add["description"] == "Add two integers."
    assert add["parameters"]["required"] == ["a"]
    assert set(add["parameters"]["properties"]) == {"a", "b"}
    assert "title" not in json.dumps(add["parameters"])
    optional = specs["lookup"]["parameters"]["properties"]["limit"]
    assert optional["type"] == "integer" and "anyOf" not in optional


def test_happy_path_and_defaults():
    out = make_registry().execute("add", '{"a": 2}')
    assert out.ok and out.content == "3"
    assert make_registry().execute("add", {"a": 2, "b": 5}).content == "7"


def test_extra_arguments_from_the_model_are_ignored():
    assert make_registry().execute("add", '{"a": 1, "b": 1, "note": "hi"}').ok


def test_invalid_json_is_reported_not_raised():
    out = make_registry().execute("add", "{oops")
    assert not out.ok and "not valid JSON" in out.content


def test_missing_and_wrongly_typed_arguments():
    registry = make_registry()
    missing = registry.execute("add", "{}")
    assert not missing.ok and "a:" in missing.content
    wrong = registry.execute("add", '{"a": "banana"}')
    assert not wrong.ok and "Invalid arguments" in wrong.content


def test_non_object_arguments():
    assert not make_registry().execute("add", "[1, 2]").ok


def test_exceptions_become_error_text():
    out = make_registry().execute("boom", "{}")
    assert not out.ok and "ZeroDivisionError" in out.content


def test_tool_error_message_is_passed_verbatim():
    out = make_registry().execute("refuse", '{"x": "q"}')
    assert not out.ok and out.content == "cannot handle 'q', try 'y'"


def test_unknown_tool_lists_alternatives():
    out = make_registry().execute("nope", "{}")
    assert not out.ok and "add" in out.content and "lookup" in out.content


def test_long_results_are_truncated_with_a_hint():
    out = make_registry().execute("big", "{}")
    assert out.ok and len(out.content) < MAX_RESULT_CHARS + 200 and "truncated" in out.content


def test_empty_arguments_string_means_no_arguments():
    out = make_registry().execute("lookup", "")
    assert not out.ok  # name is required, but it must fail as a validation error, not crash
    assert make_registry().execute("boom", "").content.startswith("ZeroDivisionError")
