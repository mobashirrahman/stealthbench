"""Apply each G03 review defect's inverse, confirm a test fails, then restore.

Run: .venv/bin/python implementation/mutation/g03_verify.py
Every mutation must FAIL the suite. A mutation that passes is a missing guard.
"""

from __future__ import annotations

import atexit
import os
import signal
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
    old: str | tuple[str, ...]
    new: str | tuple[str, ...]
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
        "M9",
        "5b: the fixture adapter's effective settings skip redaction",
        "src/stealthbench/adapters/base.py",
        """                effective_settings=redact_mapping(
                    dict(recorded.effective_settings), extra_secrets=self.extra_secrets
                ),""",
        """                effective_settings=dict(recorded.effective_settings),""",
        "tests/contract/test_provider_contract.py",
    ),
    Mutation(
        "M67",
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
        "M63",
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
        (
            '        if "stream_terminated" not in record '
            'or _was_terminated(record["stream_terminated"]):'
        ),
        '        if "stream_terminated" not in record or True:',
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
        "    if not isinstance(reason, str):\n",
        "    if not isinstance(reason, str) and reason is not None:\n",
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
        (
            "            unsupported=unsupported,"
            "\n            effective_settings=outcome.effective_settings,"
        ),
        (
            "            unsupported=_unsupported_from(recorded, request),"
            "\n            effective_settings=outcome.effective_settings,"
        ),
        "tests/contract",
    ),
    Mutation(
        "M18",
        "review-8: a malformed catalog raises out of discover()",
        "src/stealthbench/adapters/base.py",
        "        models = _catalog_models(raw)",
        "        models = list(raw.get('models', raw.get('data', [])))",
        "tests/contract",
    ),
    Mutation(
        "M19",
        "review3-9: an event index is paired with a measured total again",
        "src/stealthbench/adapters/streaming.py",
        "        return None\n\n    for index, event in enumerate(events):",
        "        return float(index)\n\n    for index, event in enumerate(events):",
        "tests/contract",
    ),
    Mutation(
        "M20",
        "review4-3: a single alias binds the unqualified key to any endpoint",
        "src/stealthbench/adapters/zen.py",
        (
            "        return self._sole_alias() == sample_key.endpoint_id"
            " and bool(sample_key.endpoint_id)"
        ),
        "        return self._sole_alias() is not None",
        "tests/contract/test_zen_adapter.py",
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
        "src/stealthbench/adapters/base.py",
        '    for key in ("context_window", "context_length", "max_context"):',
        '    for key in ("context_window", "context_length"):',
        "tests/contract",
    ),
    Mutation(
        "M24",
        "review-14: an explicit null route becomes the string 'None'",
        "src/stealthbench/adapters/base.py",
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
        "M26",
        "review3-1: the alias guard counts id only and allows zero or one",
        "src/stealthbench/adapters/zen.py",
        """        aliases = {
            entry.alias
            for entry in normalize_catalog(
                self._catalog_payload if isinstance(self._catalog_payload, Mapping) else {}
            ).entries
        }""",
        """        payload = self._catalog_payload
        models = payload.get("data", []) if isinstance(payload, Mapping) else []
        aliases = {
            item.get("id")
            for item in models
            if isinstance(item, Mapping) and item.get("id")
        }
        return len(aliases) <= 1""",
        "tests/contract/test_zen_adapter.py",
    ),
    Mutation(
        "M27",
        "review3-6: a non-True stream_terminated is read as terminated again",
        "src/stealthbench/adapters/zen.py",
        "    return read_terminated(value)",
        "    return isinstance(value, bool) and value",
        "tests/contract/test_zen_adapter.py",
    ),
    Mutation(
        "M28",
        "review3-7: a later reported count overwrites an earlier one again",
        "src/stealthbench/adapters/streaming.py",
        "            into.output_tokens if into.output_tokens is not None else newer.output_tokens",
        (
            "            newer.output_tokens"
            " if newer.output_tokens is not None else into.output_tokens"
        ),
        "tests/contract/test_streaming.py",
    ),
    Mutation(
        "M29",
        "review3-5: str.splitlines truncates a record on a unicode separator again",
        "src/stealthbench/adapters/streaming.py",
        "        for line in _sse_lines(raw_record):",
        "        for line in raw_record.splitlines():",
        "tests/contract/test_streaming.py",
    ),
    Mutation(
        "M30",
        "review3-4: an implausible pacing offset is stored instead of dropped",
        "src/stealthbench/adapters/streaming.py",
        "        streaming=_coherent_measurements(assembly),",
        "        streaming=assembly.measurements(),",
        "tests/contract/test_streaming.py",
    ),
    Mutation(
        "M31",
        "review3-2: a terminated capture with no usable frames is accepted again",
        "src/stealthbench/adapters/streaming.py",
        (
            "    if not assembly.saw_content"
            " and not (reported_output is not None and reported_output > 0):"
        ),
        "    if not assembly.saw_content:",
        "tests/contract",
    ),
    Mutation(
        "M32",
        "review3-3: the requested value of stream is fabricated as False again",
        "src/stealthbench/adapters/base.py",
        '        "stream": request.stream,',
        '        "stream": False,',
        "tests/contract",
    ),
    Mutation(
        "M33",
        "review3-13: a boolean context window is recorded on the fixture route again",
        "src/stealthbench/adapters/base.py",
        '    for key in ("context_window", "context_length", "max_context"):',
        '    for key in ("context_window", "context_length"):',
        "tests/contract/test_provider_contract.py",
    ),
    Mutation(
        "M34",
        "review3-14: an explicit null route becomes the string 'None' on the fixture route",
        "src/stealthbench/adapters/base.py",
        '    return value.strip() if isinstance(value, str) and value.strip() else "zen"',
        '    return str(value) if value is not None else "zen"',
        "tests/contract/test_provider_contract.py",
    ),
    Mutation(
        "M35",
        "review3-7: zen recorded unsupported settings are dropped on the stream path",
        "src/stealthbench/adapters/zen.py",
        (
            '                    "recorded Zen exchange carries no stream frames",\n'
            "                    extra_secrets=self.extra_secrets,\n"
            "                ),\n"
            "                unsupported=unsupported + recorded_settings,"
        ),
        (
            '                    "recorded Zen exchange carries no stream frames",\n'
            "                    extra_secrets=self.extra_secrets,\n"
            "                ),"
        ),
        "tests/contract/test_zen_adapter.py",
    ),
    Mutation(
        "M36",
        "review4-1: an empty usage block counts as content again",
        "src/stealthbench/adapters/streaming.py",
        (
            "    reported_output = assembly.usage.output_tokens\n"
            "    if not assembly.saw_content"
            " and not (reported_output is not None and reported_output > 0):"
        ),
        "    if not assembly.saw_content:",
        "tests/contract",
    ),
    Mutation(
        "M37",
        "review4-2: a truthy non-boolean stream_terminated is read as not terminated",
        "src/stealthbench/adapters/base.py",
        "        return value.strip().lower() in _TRUE_WORDS",
        "        return value.strip().lower() not in _TRUE_WORDS",
        "tests/contract",
    ),
    Mutation(
        "M38",
        "review4-3: a single alias binds the unqualified key to any endpoint",
        "src/stealthbench/adapters/zen.py",
        (
            "        return self._sole_alias() == sample_key.endpoint_id"
            " and bool(sample_key.endpoint_id)"
        ),
        "        return self._sole_alias() is not None",
        "tests/contract",
    ),
    Mutation(
        "M39",
        "review4-4: a recorded exchange fabricates a clean stop again",
        "src/stealthbench/adapters/base.py",
        """    #: Absent by default: a capture that does not say how the generation ended must
    #: not be recorded as a clean stop.
    finish_status: FinishStatus | None = None""",
        '    finish_status: FinishStatus | None = "stop"',
        "tests/contract",
    ),
    Mutation(
        "M40",
        "review4-5: an incoherent timing pair is stored whole",
        "src/stealthbench/adapters/streaming.py",
        "        and measurements.first_content_seconds > measurements.first_answer_seconds",
        "        and measurements.first_content_seconds > measurements.total_seconds",
        "tests/contract/test_streaming.py",
    ),
    Mutation(
        "M41",
        "review4-6: an alias key is dropped from the shared catalog vocabulary",
        "src/stealthbench/adapters/base.py",
        '_ALIAS_KEYS: Final[tuple[str, ...]] = ("id", "alias", "slug", "name")',
        '_ALIAS_KEYS: Final[tuple[str, ...]] = ("id", "alias", "name")',
        "tests/contract/test_provider_contract.py",
    ),
    Mutation(
        "M42",
        "review4-7: a bare string of unsupported settings becomes one per character",
        "src/stealthbench/adapters/zen.py",
        "    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):",
        "    if not isinstance(names, Sequence):",
        "tests/contract",
    ),
    Mutation(
        "M43",
        "review4-8: a reasoning-only stream is refused",
        "src/stealthbench/adapters/streaming.py",
        """        if event.reasoning_delta:
            self.saw_content = True
            self.reasoning += event.reasoning_delta""",
        """        if event.reasoning_delta:
            self.reasoning += event.reasoning_delta""",
        "tests/contract/test_streaming.py",
    ),
    Mutation(
        "M44",
        "review5-1 HIGH: a recorded failure replays as a streamed sample on the fixture route",
        "src/stealthbench/adapters/base.py",
        '        if recorded.outcome == "error":\n            # The capture records',
        "        if False:\n            # The capture records",
        "tests/contract",
    ),
    Mutation(
        "M45",
        "review5-2 HIGH: the unqualified repeat-pinned key bypasses the ambiguity guard",
        "src/stealthbench/adapters/zen.py",
        """        if allow_unqualified:
            # Both unqualified keys: a repeat-pinned one names no endpoint either.
            yield f"{sample_key.task_id}#r{sample_key.repeat_id}"
            yield sample_key.task_id""",
        """        yield f"{sample_key.task_id}#r{sample_key.repeat_id}"
        if allow_unqualified:
            yield sample_key.task_id""",
        "tests/contract/test_zen_adapter.py",
    ),
    Mutation(
        "M46",
        "review5-7: a declared secret in a reported setting is serialized verbatim",
        "src/stealthbench/adapters/base.py",
        '    return redact_mapping({"requested": value}, extra_secrets=extra_secrets)["requested"]',
        "    return value",
        "tests/contract",
    ),
    Mutation(
        "M47",
        "review5-5: the two normalizers disagree on a field again",
        "src/stealthbench/adapters/base.py",
        '    for key in ("context_window", "context_length", "max_context"):',
        '    for key in ("context_window", "context_length"):',
        "tests/contract/test_provider_contract.py",
    ),
    Mutation(
        "M48",
        "review5-6: an event index is published again as a duration in seconds",
        "src/stealthbench/adapters/streaming.py",
        """        # With no measured pacing there is no clock, and an event's position in the
        # stream is not a duration. Publishing the index in a seconds field would
        # fabricate a timing -- 0.0 for the first one, which is the substitution the
        # contract forbids.
        return None""",
        "        return float(index)",
        "tests/contract",
    ),
    Mutation(
        "M49",
        "review5-4: the streamed path drops the recorded retry hint",
        "src/stealthbench/adapters/zen.py",
        """                    # The scheduler needs the retry hint the capture recorded; dropping
                    # it here makes the streamed path retry differently from complete().
                    retry_after_seconds=_retry_after(record),""",
        "",
        "tests/contract/test_zen_adapter.py",
    ),
    Mutation(
        "M50",
        "review5-1 HIGH: a recorded failure replays as a streamed sample on the Zen route",
        "src/stealthbench/adapters/zen.py",
        '        if record.get("outcome") == "error":',
        "        if False:",
        "tests/contract/test_zen_adapter.py",
    ),
    Mutation(
        "M51",
        "review6-1 HIGH: the fixture streaming success return leaks a declared secret",
        "src/stealthbench/adapters/base.py",
        (
            "            unsupported=unsupported,"
            "\n            effective_settings=outcome.effective_settings,"
        ),
        (
            "            unsupported=_unsupported_from(recorded, request),"
            "\n            effective_settings=outcome.effective_settings,"
        ),
        "tests/contract",
    ),
    Mutation(
        "M52",
        "review6-3: the stream setting report fabricates requested=True again",
        "src/stealthbench/adapters/base.py",
        '                requested=_requested_value(request, "stream"),',
        "                requested=True,",
        "tests/contract",
    ),
    Mutation(
        "M53",
        "review6-4: zen complete() ignores a recorded failure outcome",
        "src/stealthbench/adapters/zen.py",
        (
            '        recorded_failure = record.get("outcome") == "error" or isinstance('
            '\n            record.get("failure_kind"), str\n        )'
        ),
        "        recorded_failure = False",
        "tests/contract/test_zen_adapter.py",
    ),
    Mutation(
        "M54",
        "review6-5: an incoherent capture timing raises out of complete()",
        "src/stealthbench/adapters/base.py",
        (
            "    try:\n        return StreamingMeasurements.model_validate(recorded)"
            "\n    except ValidationError:\n        return None"
        ),
        "    return StreamingMeasurements.model_validate(recorded)",
        "tests/contract/test_provider_contract.py",
    ),
    Mutation(
        "M55",
        "review6-6: the shared catalog reader prefers models over data",
        "src/stealthbench/adapters/base.py",
        '    for key in ("data", "models"):\n        value = raw.get(key)',
        '    for key in ("models", "data"):\n        value = raw.get(key)',
        "tests/contract/test_provider_contract.py",
    ),
    Mutation(
        "M56",
        "review6-7: a usage block reporting no produced tokens authorises a sample",
        "src/stealthbench/adapters/streaming.py",
        (
            "    if not assembly.saw_content"
            " and not (reported_output is not None and reported_output > 0):"
        ),
        "    if not assembly.saw_content and assembly.usage.input_tokens is None:",
        "tests/contract",
    ),
    Mutation(
        "M57",
        "review6-8: a no-fixture detail skips redaction again",
        "src/stealthbench/adapters/base.py",
        (
            "                failure=safe_failure(\n                    FailureKind.NO_FIXTURE,\n"
            "                    (\n"
            '                        f"no recorded exchange for '
            '{endpoint_id}/{benchmark_id}/{item_id} "\n'
            '                        f"repeat {sample_key.repeat_id}"\n'
            "                    ),\n"
            "                    extra_secrets=self.extra_secrets,\n"
            "                )"
        ),
        (
            "                failure=TransportFailure(\n"
            "                    kind=FailureKind.NO_FIXTURE,\n"
            "                    detail=(\n"
            '                        f"no recorded exchange for '
            '{endpoint_id}/{benchmark_id}/{item_id} "\n'
            '                        f"repeat {sample_key.repeat_id}"\n'
            "                    ),\n"
            "                )"
        ),
        "tests/contract/test_provider_contract.py",
    ),
    Mutation(
        "M58",
        "review6-2: the zen streaming refusal drops the reported settings again",
        "src/stealthbench/adapters/zen.py",
        (
            '                    "streaming was requested but this endpoint does not advertise it",'
            "\n                    extra_secrets=self.extra_secrets,\n                ),"
            "\n                unsupported=unsupported + recorded_settings,"
        ),
        (
            '                    "streaming was requested but this endpoint does not advertise it",'
            "\n                    extra_secrets=self.extra_secrets,\n                ),"
        ),
        "tests/contract/test_zen_adapter.py",
    ),
    Mutation(
        "M59",
        "coverage: a non-mapping transcript record raises out of the adapter",
        "src/stealthbench/adapters/zen.py",
        (
            "            if isinstance(record, Mapping):\n",
            "                return record\n",
        ),
        (
            "            if record is not None:\n",
            "                return record  # type: ignore[return-value]\n",
        ),
        "tests/contract",
    ),
    Mutation(
        "M60",
        "coverage: the detail reader raises on a non-mapping record",
        "src/stealthbench/adapters/zen.py",
        "    if not isinstance(record, Mapping):\n",
        "    if False:\n",
        "tests/contract",
    ),
    Mutation(
        "M61",
        "coverage: a junk top-level usage value is read as a token count",
        "src/stealthbench/adapters/zen.py",
        (
            "            if isinstance(value, bool):\n",
            "                continue\n",
            "            if isinstance(value, int) and value >= 0:\n",
        ),
        ("            if isinstance(value, (bool, int)) and (value is True or value >= 0):\n",),
        "tests/contract",
    ),
    Mutation(
        "M62",
        "coverage: a corrupt byte mid-buffer is swallowed instead of surfaced",
        "src/stealthbench/adapters/streaming.py",
        (
            "                if exc.start:\n",
            '                    yield buffer[: exc.start].decode("utf-8", errors="replace")\n',
        ),
        ("                if exc.start and False:\n",),
        "tests/contract/test_streaming.py",
    ),
    Mutation(
        "M64",
        "coverage: a stream parsed with no sample key is accepted anyway",
        "src/stealthbench/adapters/streaming.py",
        "    if sample_key is None:\n",
        "    if False:\n",
        "tests/contract/test_streaming.py",
    ),
    Mutation(
        "M65",
        "coverage: an unknown recorded failure kind is dropped instead of mapped",
        "src/stealthbench/adapters/zen.py",
        (
            "    except ValueError:\n",
            "        return FailureKind.SERVER_ERROR\n",
        ),
        (
            "    except ValueError:\n",
            "        return None\n",
        ),
        "tests/contract/test_zen_adapter.py",
    ),
    Mutation(
        "M66",
        "coverage: a usage-only stream with no produced tokens is accepted",
        "src/stealthbench/adapters/streaming.py",
        (
            "    reported_output = assembly.usage.output_tokens\n",
            "    if not assembly.saw_content"
            " and not (reported_output is not None and reported_output > 0):\n",
        ),
        ("    if not assembly.saw_content and not assembly.usage.provider_reported:\n",),
        "tests/contract",
    ),
    Mutation(
        "M68",
        "review7-1: a non-string finish reason raises out of complete()",
        "src/stealthbench/adapters/zen.py",
        "    if not isinstance(reason, str):\n",
        "    if reason is None:\n",
        "tests/contract",
    ),
    Mutation(
        "M69",
        "review7-2: an unknown recorded failure kind raises out of stream()",
        "src/stealthbench/adapters/zen.py",
        ("                    _mapped_failure_kind(recorded_kind) or FailureKind.SERVER_ERROR,\n",),
        ("                    FailureKind(recorded_kind)\n",),
        "tests/contract",
    ),
    Mutation(
        "M70",
        "review7-3: an unmapped recorded failure kind raises out of the fixture route",
        "src/stealthbench/adapters/base.py",
        (
            "    try:\n",
            "        return FailureKind(name)\n",
            "    except ValueError:\n",
        ),
        "    return FailureKind(name)\n",
        "tests/contract",
    ),
    Mutation(
        "M71",
        "review7-2: an unmapped recorded failure kind raises out of the Zen stream path",
        "src/stealthbench/adapters/zen.py",
        (
            "    try:\n",
            "        return FailureKind(name)\n",
            "    except ValueError:\n",
        ),
        "    return FailureKind(name)\n",
        "tests/contract",
    ),
    Mutation(
        "M74",
        "review7-6: the discovered snapshot cache returns an empty catalog",
        "src/stealthbench/adapters/zen.py",
        (
            "            capabilities or self._capabilities_for(sample_key.endpoint_id),\n",
            "            stream=True,\n",
        ),
        ("            self._capabilities_for(sample_key.endpoint_id),\n",),
        "tests/contract",
    ),
    Mutation(
        "M72",
        "review7: a nonsensical sentinel value is read as completion",
        "src/stealthbench/adapters/base.py",
        (
            "    if isinstance(value, int) and value in (0, 1):\n",
            "        return value == 1\n",
        ),
        (
            "    if isinstance(value, (int, float)):\n",
            "        return value != 0\n",
        ),
        "tests/contract",
    ),
    Mutation(
        "M73",
        "new: the catalog is not read from the recorded GET /models response",
        "src/stealthbench/adapters/zen.py",
        '            catalog = _catalog_from_requests(raw.get("requests"))',
        "            catalog = None",
        "tests/contract/test_g03_gate.py",
    ),
)


def _restore_on_exit(target: Path, original: str):
    """Return a callable that restores ``target``; also armed at exit and on signals."""

    def restore() -> None:
        try:
            if target.read_text(encoding="utf-8") != original:
                target.write_text(original, encoding="utf-8")
                emit(f"restored {target.relative_to(ROOT)}")
        except OSError:
            pass

    atexit.register(restore)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, _frame: (restore(), sys.exit(128 + signum)))
    return restore


def _text(value: str | tuple[str, ...]) -> str:
    """A mutation's anchor or replacement, whether written inline or as line tuples."""
    return value if isinstance(value, str) else "".join(value)


def stale_anchors() -> list[str]:
    """Mutations whose anchor no longer matches the source.

    Checked up front so a refactor that invalidates an anchor is reported once and
    loudly, instead of quietly changing what the harness measures.
    """
    stale = []
    for mutation in MUTATIONS:
        if _text(mutation.old) not in (ROOT / mutation.path).read_text(encoding="utf-8"):
            stale.append(mutation.ident)
    return stale


def run(args: list[str]) -> tuple[int, str]:
    # Bytecode caching is disabled: a cached .pyc whose mtime and size still match the
    # mutated source would make the mutation invisible, and the run would pass for the
    # wrong reason. A mutation harness must be certain it tested what it wrote.
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run(
        args, cwd=ROOT, capture_output=True, text=True, check=False, timeout=900, env=env
    )
    return proc.returncode, (proc.stdout + proc.stderr)[-1500:]


def emit(line: str) -> None:
    """Report on stderr: this script's output is evidence, not program output."""
    sys.stderr.write(f"{line}\n")
    sys.stderr.flush()


def main() -> int:
    failures: list[str] = []
    stale = stale_anchors()
    if stale:
        emit(f"STALE ANCHORS: {', '.join(stale)} -- the harness no longer matches the source")
    for mutation in MUTATIONS:
        target = ROOT / mutation.path
        original = target.read_text(encoding="utf-8")
        if _text(mutation.old) not in original:
            failures.append(f"{mutation.ident}: anchor not found in {mutation.path}")
            emit(f"{mutation.ident} SKIP  anchor missing")
            continue
        target.write_text(
            original.replace(_text(mutation.old), _text(mutation.new), mutation.count),
            encoding="utf-8",
        )
        # Restore on any exit, including an interrupt that skips the finally block:
        # a reviewer who times this script out must not be left with a mutated adapter.
        restore = _restore_on_exit(target, original)
        try:
            code, out = run(
                [PY, "-m", "pytest", "--strict-markers", "-q", "-x", *mutation.tests.split()]
            )
        finally:
            restore()
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
