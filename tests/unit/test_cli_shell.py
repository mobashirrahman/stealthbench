"""Unit tests for the CLI shell internals (task T00A)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stealthbench import SCHEMA_VERSION, __version__
from stealthbench.cli import (
    EXIT_ERROR,
    EXIT_NOT_IMPLEMENTED,
    EXIT_OK,
    EXIT_USAGE,
    GatePending,
    UserError,
    _existing_file,
    build_parser,
    main,
)
from tests.conftest import REPO_ROOT

#: Commands the plan declares, and the gate each one belongs to.
#: Commands still awaiting their gate. `manifest validate` left this set in G01.
#: Commands that are declared but not yet implemented. ``run`` left this set in G04: it
#: validates its input, resolves the mode, and refuses an unauthorized live run, but the
#: offline dispatch loop lands with the vertical workflow in G05. It now exits 3 with the
#: same payload shape rather than 0, because exit 0 would read as a completed campaign.
PENDING_COMMANDS = {
    ("replay",): "G02",
    ("report",): "G13",
    ("signatures",): "G11",
    ("identify",): "G12",
}


def test_exit_codes_are_distinct_and_non_zero_for_pending_work() -> None:
    codes = [EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_NOT_IMPLEMENTED]
    assert len(set(codes)) == len(codes)
    assert EXIT_OK == 0
    assert all(code != 0 for code in codes[1:])


def test_gate_pending_names_the_command_and_gate() -> None:
    error = GatePending("replay", "G02")
    assert error.command == "replay"
    assert error.gate == "G02"
    assert "replay" in str(error)
    assert "G02" in str(error)


def test_parser_requires_a_subcommand() -> None:
    """No command must be a usage error, so the CLI can never exit 0 having done nothing."""
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args([])
    assert excinfo.value.code == EXIT_USAGE


def test_parser_rejects_an_unknown_command() -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["not-a-command"])
    assert excinfo.value.code == EXIT_USAGE


def test_parser_accepts_every_declared_command(tmp_path: Path) -> None:
    manifest = tmp_path / "campaign.json"
    manifest.write_text("{}", encoding="utf-8")
    parser = build_parser()
    for argv_prefix in [*PENDING_COMMANDS, ("run",)]:
        extra = ["--output", str(tmp_path / "out")] if argv_prefix == ("report",) else []
        argv = [*argv_prefix, str(manifest), *extra]
        assert parser.parse_args(argv).command == argv_prefix[0]


@pytest.mark.parametrize(
    ("argv_prefix", "expected_command"),
    [(key, " ".join(key)) for key in PENDING_COMMANDS],
)
def test_pending_command_names_its_gate(
    argv_prefix: tuple[str, ...],
    expected_command: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each unimplemented command must name the gate that will implement it."""
    manifest = tmp_path / "campaign.json"
    manifest.write_text("{}", encoding="utf-8")
    extra: list[str] = []
    if argv_prefix == ("report",):
        extra = ["--output", str(tmp_path / "out")]

    assert main([*argv_prefix, str(manifest), *extra]) == EXIT_NOT_IMPLEMENTED

    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err.splitlines()[-1])
    assert payload["status"] == "not_implemented"
    assert payload["command"] == expected_command
    assert payload["gate"] == PENDING_COMMANDS[argv_prefix]
    assert not (tmp_path / "out").exists(), "a pending command must not write artifacts"


def test_existing_file_rejects_a_missing_path(tmp_path: Path) -> None:
    with pytest.raises(UserError, match="not a file"):
        _existing_file(str(tmp_path / "absent.json"))


def test_existing_file_rejects_a_directory(tmp_path: Path) -> None:
    with pytest.raises(UserError, match="not a file"):
        _existing_file(str(tmp_path))


def test_existing_file_accepts_a_real_file(tmp_path: Path) -> None:
    target = tmp_path / "campaign.json"
    target.write_text(json.dumps({"schema_version": SCHEMA_VERSION}), encoding="utf-8")
    assert _existing_file(str(target)) == target


def test_version_is_a_non_empty_dotted_string() -> None:
    assert __version__
    assert __version__.count(".") >= 1
    assert all(part.isdigit() for part in __version__.split("."))


def test_main_defaults_to_system_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no argv argument the CLI reads sys.argv, as the console script does."""
    monkeypatch.setattr("sys.argv", ["stealthbench", "version"])
    assert main() == EXIT_OK
    assert json.loads(capsys.readouterr().out)["version"] == __version__


def test_main_writes_to_the_supplied_stream(tmp_path: Path) -> None:
    sink = tmp_path / "out.txt"
    with sink.open("w") as handle:
        assert main(["version"], stdout=handle) == EXIT_OK
    assert json.loads(sink.read_text(encoding="utf-8"))["version"] == __version__


def test_pending_command_writes_status_to_stderr_not_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "campaign.json"
    manifest.write_text("{}", encoding="utf-8")
    assert main(["replay", str(manifest)]) == EXIT_NOT_IMPLEMENTED
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err.splitlines()[-1]) == {
        "command": "replay",
        "gate": "G02",
        "status": "not_implemented",
    }


def test_cli_shell_imports_no_third_party_dependency() -> None:
    """`--help` and `version` must work on a bare install with no dependencies.

    Only module-scope imports are considered: the schema layer is imported inside
    the manifest handler, so the shell itself has no import-time dependency beyond
    the standard library. That is what keeps a fresh `--no-deps` install a
    meaningful smoke test.
    """
    import ast
    import sys

    module = sys.modules["stealthbench.cli"]
    assert module.__file__ is not None
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))

    imported: set[str] = set()
    for node in tree.body:  # module scope only, not function bodies
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])

    external = {
        name
        for name in imported
        if name not in sys.stdlib_module_names and not name.startswith("stealthbench")
    }
    assert external == set(), f"module-scope third-party imports: {sorted(external)}"


def test_schema_layer_is_imported_lazily_inside_the_manifest_handler() -> None:
    """Guard the property the previous test depends on."""
    import ast
    import sys

    module = sys.modules["stealthbench.cli"]
    assert module.__file__ is not None
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    handler = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "cmd_manifest_validate"
    )
    imported_inside: set[str] = set()
    for node in ast.walk(handler):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_inside.add(node.module.split(".")[0])
    assert "pydantic" in imported_inside
    assert "stealthbench" in imported_inside


def _offline_manifest(*, campaign_id: str) -> dict:
    """A schema-valid offline manifest, derived from the committed demo profile.

    Reusing the shipped file rather than hand-rolling one keeps these tests honest: a
    hand-written manifest could drift from the schema without failing here.
    """
    payload = json.loads((REPO_ROOT / "configs" / "offline-demo.json").read_text(encoding="utf-8"))
    payload["campaign_id"] = campaign_id
    payload.pop("authorization", None)
    return payload


def test_run_refuses_an_unauthorized_live_run_without_dispatching(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``run`` landed in G04: it refuses, and names every missing requirement.

    The refusal must not look like a completed campaign, so it exits with the usage
    code and reports zero dispatches.
    """
    manifest = tmp_path / "campaign.json"
    manifest.write_text(
        json.dumps(_offline_manifest(campaign_id="c-live-refused")), encoding="utf-8"
    )
    assert main(["run", str(manifest), "--live"]) == EXIT_USAGE

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "refused"
    assert payload["dispatched"] == 0
    assert payload["blockers"], "a refusal must say what is missing, not just that it is missing"


def test_run_offline_exits_not_implemented_rather_than_succeeding(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Offline ``run`` has no dispatch loop yet and must not exit 0.

    A zero exit is the one thing in this output a calling script cannot misread, so an
    undispatched campaign has to fail rather than look finished.
    """
    manifest = tmp_path / "campaign.json"
    manifest.write_text(json.dumps(_offline_manifest(campaign_id="c-offline")), encoding="utf-8")
    assert main(["run", str(manifest)]) == EXIT_NOT_IMPLEMENTED

    captured = capsys.readouterr()
    payload = json.loads(captured.err.splitlines()[-1])
    assert payload["status"] == "not_implemented"
    assert payload["command"] == "run"
    assert payload["gate"] == "G05"
    assert captured.out == "", "a pending run must not write a result to stdout"
