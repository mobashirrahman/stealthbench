"""Apply each G03 review defect's inverse, confirm a test fails, then restore.

Run: .venv/bin/python implementation/mutation/g03_verify.py
Every mutation must FAIL the suite. A mutation that passes is a missing guard.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / ".venv" / "bin" / "python")


@dataclass(frozen=True)
class Mutation:
    ident: str
    defect: str
    path: str
    old: str
    new: str
    tests: str
    count: int = 1


MUTATIONS = (
    Mutation(
        "M1",
        "1: terminal finish_reason dropped when content shares the record",
        "src/stealthbench/adapters/streaming.py",
        "            content_delta=content_delta,\n"
        "            reasoning_delta=reasoning_delta,\n"
        "            finish_reason=finish_reason,",
        "            content_delta=content_delta,\n            reasoning_delta=reasoning_delta,",
        "tests/contract/test_streaming.py",
    ),
    Mutation(
        "M2",
        "2: CRLF/CR framing loses all content",
        "src/stealthbench/adapters/streaming.py",
        'self._buffer += chunk.replace("\\r\\n", "\\n").replace("\\r", "\\n")',
        "self._buffer += chunk",
        "tests/contract/test_streaming.py tests/contract/test_g03_gate.py",
    ),
    Mutation(
        "M3",
        "3: a string/number capability becomes True",
        "src/stealthbench/adapters/base.py",
        """    for value in candidates:
        if isinstance(value, bool):
            return value
    return False""",
        """    for value in candidates:
        if value is not None:
            return bool(value)
    return False""",
        "tests/contract/test_provider_contract.py",
    ),
    Mutation(
        "M4",
        "4: junk token usage coerced instead of absent",
        "src/stealthbench/adapters/base.py",
        """    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None""",
        """    if value is None:
        return None
    return int(value)""",
        "tests/contract/test_provider_contract.py",
    ),
    Mutation(
        "M5",
        "5a: Zen effective settings skip redaction entirely",
        "src/stealthbench/adapters/zen.py",
        "    effective: dict[str, Any] = {",
        "    extra_secrets = frozenset()\n    effective: dict[str, Any] = {",
        "tests/contract/test_zen_adapter.py",
    ),
    Mutation(
        "M5b",
        "5b: the fixture adapter's effective settings skip redaction",
        "src/stealthbench/adapters/base.py",
        """                effective_settings=redact_mapping(
                    dict(recorded.effective_settings), extra_secrets=self.extra_secrets
                ),""",
        """                effective_settings=dict(recorded.effective_settings),""",
        "tests/contract/test_provider_contract.py",
    ),
    Mutation(
        "M5c",
        "5c: a fixture tool-call argument skips redaction",
        "src/stealthbench/adapters/base.py",
        """                    "tool_calls": redact_mapping(
                        {"calls": [dict(call) for call in recorded.tool_calls]},
                        extra_secrets=self.extra_secrets,
                    )["calls"],""",
        """                    "tool_calls": [dict(call) for call in recorded.tool_calls],""",
        "tests/contract/test_provider_contract.py",
    ),
    Mutation(
        "M5d",
        "5d: Zen provider metadata skips redaction",
        "src/stealthbench/adapters/zen.py",
        "redacted_provider_metadata=redact_mapping(metadata, extra_secrets=self.extra_secrets)",
        "redacted_provider_metadata=metadata",
        "tests/contract/test_zen_adapter.py",
    ),
    Mutation(
        "M6",
        "6: streamed results carry no route label",
        "src/stealthbench/adapters/streaming.py",
        '            "route": route,',
        '            "route": "unlabeled",',
        "tests/contract/test_streaming.py",
    ),
    Mutation(
        "M7",
        "8: the generic repeat record shadows a repeat-specific one",
        "src/stealthbench/adapters/zen.py",
        "        for candidate in self._candidate_keys("
        "sample_key, allow_unqualified=allow_unqualified):",
        "        for candidate in reversed(tuple(self._candidate_keys(sample_key))):",
        "tests/contract/test_zen_adapter.py",
    ),
    Mutation(
        "M8",
        "new: mid-stream error frame silently becomes an interruption",
        "src/stealthbench/adapters/streaming.py",
        "    if assembly.error_message is not None:",
        "    if False:",
        "tests/contract/test_streaming.py tests/contract/test_g03_gate.py",
    ),
    Mutation(
        "M10",
        "review-1 CRITICAL: the fixture path synthesises data: [DONE] again",
        "src/stealthbench/adapters/base.py",
        ') + (b"data: [DONE]\\n\\n" if recorded.stream_terminated else b"")',
        ') + b"data: [DONE]\\n\\n"',
        "tests/contract",
    ),
    Mutation(
        "M11",
        "review-1 CRITICAL: the Zen path synthesises data: [DONE] again",
        "src/stealthbench/adapters/zen.py",
        '        if record.get("stream_terminated", True) is not False:',
        "        if True:",
        "tests/contract",
    ),
    Mutation(
        "M12",
        "review-2 MAJOR: declared secrets are not forwarded to the streaming parser",
        "src/stealthbench/adapters/base.py",
        '            adapter="fixture",\n            extra_secrets=self.extra_secrets,',
        '            adapter="fixture",',
        "tests/contract",
    ),
    Mutation(
        "M13",
        "review-3 MAJOR: an absent finish reason is reported as a clean stop",
        "src/stealthbench/adapters/zen.py",
        "    if reason is None:\n        return None",
        '    if reason is None:\n        return "stop"',
        "tests/contract",
    ),
    Mutation(
        "M14",
        "review-4 MAJOR: an eagerly converted trailing CR fabricates a record boundary",
        "src/stealthbench/adapters/streaming.py",
        """        if chunk.endswith("\\r"):
            self._pending_cr = True
            chunk = chunk[:-1]""",
        '        if chunk.endswith("\\r"):\n            chunk = chunk[:-1] + chr(10)',
        "tests/contract",
    ),
    Mutation(
        "M15",
        "review-5: usage on the terminal record is dropped again",
        "src/stealthbench/adapters/streaming.py",
        """            kind=StreamEventKind.FINISH,
            finish_reason=finish_reason,
            usage=usage_from_response(payload) if has_usage else Usage(),""",
        """            kind=StreamEventKind.FINISH,
            finish_reason=finish_reason,""",
        "tests/contract",
    ),
    Mutation(
        "M16",
        "review-6: a content-parts delta is dropped again",
        "src/stealthbench/adapters/streaming.py",
        (
            "            elif isinstance(content, Sequence) "
            "and not isinstance(content, (str, bytes)):"
        ),
        """            elif False:""",
        "tests/contract",
    ),
    Mutation(
        "M17",
        "review-7: recorded unsupported settings are ignored when streaming",
        "src/stealthbench/adapters/base.py",
        "            unsupported=_unsupported_from(recorded, request),",
        "            unsupported=(),",
        "tests/contract",
    ),
    Mutation(
        "M18",
        "review-8: a malformed catalog raises out of discover()",
        "src/stealthbench/adapters/base.py",
        """        if not isinstance(models, Sequence) or isinstance(models, (str, bytes)):
            # A malformed catalog is an empty observation, not an exception: discovery
            # must never take the campaign down on one bad fixture.
            models = []""",
        "        models = list(models)",
        "tests/contract",
    ),
    Mutation(
        "M19",
        "review-9: a fabricated offset is paired with a measured total again",
        "src/stealthbench/adapters/streaming.py",
        """        if total_seconds is not None:
            return None
        return float(index)""",
        """        return float(index)""",
        "tests/contract",
    ),
    Mutation(
        "M20",
        "review-10: an unqualified transcript key is matched however many aliases exist",
        "src/stealthbench/adapters/zen.py",
        """        yield f"{sample_key.endpoint_id}:{sample_key.task_id}"
        if allow_unqualified:
            yield sample_key.task_id""",
        """        yield f"{sample_key.endpoint_id}:{sample_key.task_id}"
        yield sample_key.task_id""",
        "tests/contract",
    ),
    Mutation(
        "M21",
        "review-11: a corrupt byte stalls the rest of the stream again",
        "src/stealthbench/adapters/streaming.py",
        """                if exc.end == len(buffer):
                    # The failing sequence runs to the end of what we have, so it may
                    # simply be truncated: hold it back for the next read.
                    decoder_partial = buffer
                    break""",
        """                if exc.end == len(buffer) or exc.start == 0:
                    decoder_partial = buffer
                    break""",
        "tests/contract",
    ),
    Mutation(
        "M22",
        "review-12: a /models path with a query string is not recognised",
        "src/stealthbench/adapters/zen.py",
        '        bare = path.split("?", 1)[0].rstrip("/")',
        '        bare = path.rstrip("/")',
        "tests/contract",
    ),
    Mutation(
        "M23",
        "review-13: a boolean context window is recorded as a measurement",
        "src/stealthbench/adapters/zen.py",
        "if isinstance(context, int) and not isinstance(context, bool) and context > 0",
        "if isinstance(context, int) and context > 0",
        "tests/contract",
    ),
    Mutation(
        "M24",
        "review-14: an explicit null route becomes the string 'None'",
        "src/stealthbench/adapters/zen.py",
        '    return value.strip() if isinstance(value, str) and value.strip() else "zen"',
        '    return str(value) if value is not None else "zen"',
        "tests/contract",
    ),
    Mutation(
        "M25",
        "review-15: the two catalog digests diverge again",
        "src/stealthbench/adapters/zen.py",
        "    return snapshot.digest()",
        '    return content_digest({"source": snapshot.source})',
        "tests/contract",
    ),
    Mutation(
        "M9",
        "new: the catalog is not read from the recorded GET /models response",
        "src/stealthbench/adapters/zen.py",
        '            catalog = _catalog_from_requests(raw.get("requests"))',
        "            catalog = None",
        "tests/contract/test_g03_gate.py",
    ),
)


def run(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False, timeout=900)
    return proc.returncode, (proc.stdout + proc.stderr)[-1500:]


def emit(line: str) -> None:
    """Report on stderr: this script's output is evidence, not program output."""
    sys.stderr.write(f"{line}\n")
    sys.stderr.flush()


def main() -> int:
    failures: list[str] = []
    for mutation in MUTATIONS:
        target = ROOT / mutation.path
        original = target.read_text(encoding="utf-8")
        if mutation.old not in original:
            failures.append(f"{mutation.ident}: anchor not found in {mutation.path}")
            emit(f"{mutation.ident} SKIP  anchor missing")
            continue
        target.write_text(
            original.replace(mutation.old, mutation.new, mutation.count), encoding="utf-8"
        )
        try:
            code, out = run(
                [PY, "-m", "pytest", "--strict-markers", "-q", "-x", *mutation.tests.split()]
            )
        finally:
            target.write_text(original, encoding="utf-8")
        if code != 0:
            emit(f"{mutation.ident} PASS  caught ({mutation.defect})")
        else:
            failures.append(f"{mutation.ident}: mutation survived ({mutation.defect})")
            emit(f"{mutation.ident} FAIL  SURVIVED -- no test caught it")
            emit(out[-600:])

    code, out = run([PY, "-m", "pytest", "--strict-markers", "-q", "tests/contract"])
    if code != 0:
        failures.append("restore check: contract suite does not pass after restore")
        emit("restore FAIL")
        emit(out[-800:])
    else:
        emit("restore PASS  contract suite green after all restores")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
