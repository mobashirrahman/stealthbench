"""G11 T11A: probe bank, baseline and tokenizer inventory (unit)."""

from __future__ import annotations

import pytest

from stealthbench.fingerprints.probes import (
    BASELINE_TEXT,
    DEFAULT_TOKENIZER_ASSETS,
    FRAMING_USER_PREFIX,
    FRAMING_USER_SUFFIX,
    PROBE_BANK_VERSION,
    PROBE_COUNT,
    PROBE_REPEATS,
    TOKENIZER_CAVEAT,
    TokenizerAsset,
    baseline_framed_message,
    build_probe_bank,
    framed_message,
    get_probe,
    probe_bank_digest,
    probe_bank_manifest,
    resolve_tokenizer_asset,
    validate_probe_bank,
    validate_tokenizer_assets,
)

pytestmark = pytest.mark.unit


def test_bank_holds_sixty_versioned_probes_with_three_repeats() -> None:
    bank = build_probe_bank()
    assert len(bank) == 60
    assert PROBE_COUNT == 60
    assert PROBE_REPEATS == 3
    assert PROBE_BANK_VERSION == "probe-v1"
    manifest = dict(probe_bank_manifest())
    assert manifest["probe_version"] == "probe-v1"
    assert manifest["probe_count"] == 60
    assert manifest["repeats"] == 3
    assert manifest["separate_from_benchmarks"] is True
    assert manifest["tokenizer_caveat"] == TOKENIZER_CAVEAT


def test_probe_ids_are_unique_and_ordered() -> None:
    bank = build_probe_bank()
    ids = [probe.probe_id for probe in bank]
    assert ids == [f"sb-fp-{index:03d}" for index in range(1, 61)]
    assert len(set(ids)) == 60
    validate_probe_bank(bank)


def test_bank_covers_every_required_content_family() -> None:
    bank = build_probe_bank()
    categories = {probe.category for probe in bank}
    assert {
        "code",
        "whitespace",
        "punctuation",
        "unicode",
        "emoji",
        "multilingual",
        "numeric",
        "boundary_pair",
        "behavior_format",
        "behavior_task",
    } <= categories


def test_boundary_pairs_include_concatenations() -> None:
    bank = build_probe_bank()
    by_id = {probe.probe_id: probe for probe in bank}
    # sb-fp-033/034 join into sb-fp-035; same shape repeats for the other triples.
    assert by_id["sb-fp-033"].text + by_id["sb-fp-034"].text == by_id["sb-fp-035"].text
    assert by_id["sb-fp-036"].text + by_id["sb-fp-037"].text == by_id["sb-fp-038"].text
    assert by_id["sb-fp-039"].text + by_id["sb-fp-040"].text == by_id["sb-fp-041"].text


def test_baseline_and_framing_are_fixed() -> None:
    assert BASELINE_TEXT == "The quick brown fox jumps over the lazy dog."
    first = baseline_framed_message()
    assert BASELINE_TEXT in first
    assert first.startswith(FRAMING_USER_PREFIX)
    assert first.endswith(FRAMING_USER_SUFFIX)
    # Deterministic: rebuilding the bank never moves the framing.
    assert baseline_framed_message() == first
    bank = build_probe_bank()
    probe = get_probe(bank, "sb-fp-001")
    message = framed_message(probe)
    assert probe.text in message
    assert message.startswith(FRAMING_USER_PREFIX)
    assert message.endswith(FRAMING_USER_SUFFIX)
    assert framed_message(probe) == message


def test_token_count_probes_keep_outputs_short() -> None:
    bank = build_probe_bank()
    for probe in bank:
        if probe.signal == "input_count_vector":
            assert probe.max_output_tokens <= 16


def test_get_probe_rejects_unknown_ids() -> None:
    bank = build_probe_bank()
    with pytest.raises(KeyError):
        get_probe(bank, "sb-fp-999")


def test_tokenizer_assets_pin_revisions_without_remote_code() -> None:
    assert len(DEFAULT_TOKENIZER_ASSETS) >= 1
    validate_tokenizer_assets(DEFAULT_TOKENIZER_ASSETS)
    for asset in DEFAULT_TOKENIZER_ASSETS:
        assert asset.revision
        assert asset.requires_remote_code is False
        assert resolve_tokenizer_asset(asset) == asset


def test_tokenizer_resolution_refuses_remote_executable_code() -> None:
    asset = DEFAULT_TOKENIZER_ASSETS[0]
    with pytest.raises(ValueError, match="remote executable code is disabled"):
        resolve_tokenizer_asset(asset, allow_remote_code=True)
    remote_asset = TokenizerAsset(
        name="evil-tokenizer",
        revision="rev-1",
        source="remote",
        requires_remote_code=True,
    )
    with pytest.raises(ValueError, match="remote executable code"):
        resolve_tokenizer_asset(remote_asset)
    with pytest.raises(ValueError, match="remote executable code"):
        validate_tokenizer_assets([remote_asset])


def test_manifest_is_reproducible() -> None:
    first = probe_bank_manifest()
    second = probe_bank_manifest(build_probe_bank(), DEFAULT_TOKENIZER_ASSETS)
    assert probe_bank_digest(dict(first)) == probe_bank_digest(dict(second))
    # Canonical form: key order in memory does not move the digest.
    shuffled = dict(reversed(list(dict(first).items())))
    assert probe_bank_digest(shuffled) == probe_bank_digest(dict(first))


def test_manifest_changes_when_content_or_library_changes() -> None:
    from stealthbench.fingerprints.probes import Probe

    base_digest = probe_bank_digest()
    bank = build_probe_bank()
    changed = tuple(
        Probe(
            probe_id=probe.probe_id,
            category=probe.category,
            signal=probe.signal,
            text=probe.text + "!",
            max_output_tokens=probe.max_output_tokens,
            purpose=probe.purpose,
        )
        if probe.probe_id == "sb-fp-001"
        else probe
        for probe in bank
    )
    assert probe_bank_digest(dict(probe_bank_manifest(changed))) != base_digest
    relabeled = (
        TokenizerAsset(
            name=DEFAULT_TOKENIZER_ASSETS[0].name,
            revision="rev-CHANGED",
            source=DEFAULT_TOKENIZER_ASSETS[0].source,
        ),
        *DEFAULT_TOKENIZER_ASSETS[1:],
    )
    assert probe_bank_digest(dict(probe_bank_manifest(None, relabeled))) != base_digest


def test_validation_rejects_bad_banks() -> None:
    bank = list(build_probe_bank())
    with pytest.raises(ValueError, match="must hold 60"):
        validate_probe_bank(bank[:59])
    duplicated = [*bank]
    duplicated[0] = duplicated[1]
    with pytest.raises(ValueError, match="unique"):
        validate_probe_bank(tuple(duplicated))
