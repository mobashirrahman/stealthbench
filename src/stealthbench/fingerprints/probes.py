"""Versioned 60-probe signature bank (G11 T11A).

Contracts (``BENCHMARK_PLAN.md`` "Observable signatures", ``docs/contracts.md``
``SignatureResult``/``IdentityReport``):

* 60 signature prompts, repeated 3 times, separate from benchmarks and
  classifier holdouts. Short outputs for token-count probes to control cost.
* Fixed message framing. Tokenizer evidence comes from deltas::

      delta(text) = input_tokens(fixed message containing text)
                    - input_tokens(fixed baseline message)

  A fixed framing offset carries no identity information and is removed by
  the subtraction (see :mod:`stealthbench.fingerprints.features`).
* Texts cover code, whitespace, punctuation, Unicode, emoji, multilingual
  content and numeric strings, plus paired strings and concatenations that
  expose token boundary effects, plus fixed harmless behavior tasks.
* Tokenizer caveat (arXiv 2608.29930): an exact count-vector match supports
  a shared tokenization stack. It cannot establish identical weights:
  different models can share a tokenizer and related models can use
  different serving templates. Every consumer must repeat that caveat
  rather than derive an identification threshold from it.
* Tokenizer assets are inventory records with pinned revisions. Resolving
  an asset never executes remote code (``trust_remote_code``-style
  behaviour is refused outright).
* Manifests are reproducible: the same bank always serializes to the same
  canonical digest.

Offline by construction: pure data and pure functions. No transport,
no socket, no subprocess, no credential.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

from stealthbench.schemas.hashing import content_digest

#: Probe bank version. Pinned in ``configs/pilot.json`` and ``full.json``
#: (``signature_probes.probe_version``); changing any probe text, the
#: baseline, or the framing requires a version bump.
PROBE_BANK_VERSION: Final[str] = "probe-v1"

#: Exactly 60 probes per the benchmark plan.
PROBE_COUNT: Final[int] = 60

#: Every probe is collected 3 times (``signature_probes.repeats``).
PROBE_REPEATS: Final[int] = 3

#: Fixed baseline text. Deltas are measured against the framed form of this
#: string (see :func:`baseline_framed_message`).
BASELINE_TEXT: Final[str] = "The quick brown fox jumps over the lazy dog."

#: Fixed framing around every probe text. Constant across the whole bank so
#: a constant overhead subtracts out of every delta.
FRAMING_SYSTEM_PROMPT: Final[str] = "You are a helpful assistant. Follow the instruction exactly."
FRAMING_USER_PREFIX: Final[str] = "Complete the following:\n"
FRAMING_USER_SUFFIX: Final[str] = "\nKeep the answer short."

#: Tokenizer caveat repeated wherever a count-vector match is reported.
TOKENIZER_CAVEAT: Final[str] = (
    "An exact count-vector match supports a shared tokenization stack. "
    "It cannot establish identical weights: different models can share a "
    "tokenizer, and related models can use different serving templates "
    "(arXiv 2608.29930)."
)

ProbeSignal = Literal["input_count_vector", "behavior"]


@dataclass(frozen=True, slots=True)
class Probe:
    """One signature probe: a fixed text collected under fixed framing."""

    probe_id: str
    category: str
    signal: ProbeSignal
    text: str
    max_output_tokens: int
    purpose: str


@dataclass(frozen=True, slots=True)
class TokenizerAsset:
    """A versioned public tokenizer asset used as a comparison reference.

    An inventory record, not a download. ``revision`` pins the asset
    (commit, tag, or digest). ``requires_remote_code`` must be ``False``
    for every default asset: resolving executable tokenizer code from the
    network is refused (see :func:`resolve_tokenizer_asset`).
    """

    name: str
    revision: str
    source: str
    requires_remote_code: bool = False


#: Reference tokenizer inventory. Revisions are recorded so a changed
#: library is detectable; none of these require remote executable code.
DEFAULT_TOKENIZER_ASSETS: Final[tuple[TokenizerAsset, ...]] = (
    TokenizerAsset(
        name="local-bpe-reference-a",
        revision="rev-2026-09-01-a1b2c3",
        source="local",
    ),
    TokenizerAsset(
        name="local-bpe-reference-b",
        revision="rev-2026-09-01-d4e5f6",
        source="local",
    ),
    TokenizerAsset(
        name="local-sentencepiece-reference-c",
        revision="rev-2026-09-01-789abc",
        source="local",
    ),
)

# ---------------------------------------------------------------------------
# The 60 fixed probe texts.
#
# Each entry: (category, signal, text, max_output_tokens, purpose).
# IDs are sb-fp-001 .. sb-fp-060 in list order. Texts are literals so the
# bank is reproducible without a seed. Short outputs keep costs bounded.
# ---------------------------------------------------------------------------

_PROBE_SPECS: Final[tuple[tuple[str, ProbeSignal, str, int, str], ...]] = (
    (
        "code",
        "input_count_vector",
        "def add(a, b):\n    return a + b",
        16,
        "short python function; indentation-sensitive tokenization",
    ),
    (
        "code",
        "input_count_vector",
        "for i in range(10):\n    print(f'item {i}')",
        16,
        "f-string loop; quotes and braces",
    ),
    (
        "code",
        "input_count_vector",
        "x = [i*i for i in range(8) if i % 2 == 0]",
        16,
        "list comprehension; operators and brackets",
    ),
    (
        "code",
        "input_count_vector",
        '{\n  "name": "ada",\n  "scores": [1, 2, 3]\n}',
        16,
        "json literal; quotes, colons, brackets",
    ),
    (
        "code",
        "input_count_vector",
        "SELECT id, name FROM users WHERE age > 21;",
        16,
        "sql line; keywords and comparison",
    ),
    (
        "code",
        "input_count_vector",
        'fn main() {\n    println!("hi");\n}',
        16,
        "rust snippet; macro punctuation",
    ),
    (
        "code",
        "input_count_vector",
        "    indented with four spaces\n\tindented with a tab",
        16,
        "mixed indentation; spaces versus tab",
    ),
    (
        "code",
        "input_count_vector",
        "a==b and c!=d or not e",
        16,
        "boolean operators; multi-char symbols",
    ),
    (
        "whitespace",
        "input_count_vector",
        "a b  c   d",
        16,
        "multiple spaces; run-length sensitivity",
    ),
    (
        "whitespace",
        "input_count_vector",
        "line one\nline two\n\nline four",
        16,
        "blank line; newline runs",
    ),
    ("whitespace", "input_count_vector", "col1\tcol2\tcol3", 16, "tabs as separators"),
    (
        "whitespace",
        "input_count_vector",
        "trailing space here ",
        16,
        "trailing space; edge trimming",
    ),
    (
        "punctuation",
        "input_count_vector",
        "Hello, world! How are you?",
        16,
        "common punctuation cluster",
    ),
    ("punctuation", "input_count_vector", "(a+b)*[c-d]/{e:f};", 16, "dense bracket mix"),
    ("punctuation", "input_count_vector", "... --- ... !!! ???", 16, "repeated punctuation runs"),
    ("punctuation", "input_count_vector", "\"quoted\" 'single' `tick`", 16, "quote-style variants"),
    (
        "unicode",
        "input_count_vector",
        "caf\u00e9 na\u00efve r\u00e9sum\u00e9",
        16,
        "latin diacritics; composed forms",
    ),
    (
        "unicode",
        "input_count_vector",
        "e\u0301 combined versus \u00e9 precomposed",
        16,
        "combining mark versus precomposed",
    ),
    (
        "unicode",
        "input_count_vector",
        "\u200bzero-width\u200b inside",
        16,
        "zero-width spaces; invisible segmentation",
    ),
    (
        "unicode",
        "input_count_vector",
        "\uff21\uff22\uff23 fullwidth latin",
        16,
        "fullwidth codepoints",
    ),
    (
        "emoji",
        "input_count_vector",
        "\U0001f600\U0001f601\U0001f602",
        16,
        "emoji run; multi-byte sequence",
    ),
    (
        "emoji",
        "input_count_vector",
        "\U0001f469\u200d\U0001f4bb technologist sequence",
        16,
        "zwj emoji sequence; joiner handling",
    ),
    (
        "multilingual",
        "input_count_vector",
        "\u4f60\u597d\u4e16\u754c",
        16,
        "chinese; cjk segmentation",
    ),
    ("multilingual", "input_count_vector", "مرحبا بالعالم", 16, "arabic; right-to-left script"),
    ("multilingual", "input_count_vector", "नमस्ते दुनिया", 16, "hindi devanagari; conjuncts"),
    (
        "multilingual",
        "input_count_vector",
        "Hola, \u00bfc\u00f3mo est\u00e1s?",
        16,
        "spanish; inverted marks",
    ),
    ("multilingual", "input_count_vector", "Привет мир", 16, "russian cyrillic"),
    (
        "multilingual",
        "input_count_vector",
        "\u3053\u3093\u306b\u3061\u306f\u4e16\u754c",
        16,
        "japanese hiragana plus kanji",
    ),
    ("numeric", "input_count_vector", "1234567890", 16, "digit run; number chunking"),
    ("numeric", "input_count_vector", "3.14159265358979", 16, "decimal; dot segmentation"),
    ("numeric", "input_count_vector", "0xDEADBEEF cafef00d", 16, "hex tokens; letter-digit mix"),
    (
        "numeric",
        "input_count_vector",
        "1000000 vs 1,000,000 vs 1000000000",
        16,
        "grouping commas; magnitude phrasing",
    ),
    (
        "boundary_pair",
        "input_count_vector",
        "token",
        16,
        "boundary half A; pairs with sb-fp-034/035",
    ),
    (
        "boundary_pair",
        "input_count_vector",
        "ization",
        16,
        "boundary half B; pairs with sb-fp-033/035",
    ),
    (
        "boundary_pair",
        "input_count_vector",
        "tokenization",
        16,
        "boundary concatenation of sb-fp-033 + sb-fp-034",
    ),
    ("boundary_pair", "input_count_vector", "un", 16, "boundary half A; pairs with sb-fp-037/038"),
    (
        "boundary_pair",
        "input_count_vector",
        "happy",
        16,
        "boundary half B; pairs with sb-fp-036/038",
    ),
    (
        "boundary_pair",
        "input_count_vector",
        "unhappy",
        16,
        "boundary concatenation of sb-fp-036 + sb-fp-037",
    ),
    (
        "boundary_pair",
        "input_count_vector",
        "data",
        16,
        "boundary half A; pairs with sb-fp-040/041",
    ),
    (
        "boundary_pair",
        "input_count_vector",
        "base",
        16,
        "boundary half B; pairs with sb-fp-039/041",
    ),
    (
        "boundary_pair",
        "input_count_vector",
        "database",
        16,
        "boundary concatenation of sb-fp-039 + sb-fp-040",
    ),
    (
        "boundary_pair",
        "input_count_vector",
        "super",
        16,
        "boundary half A; pairs with sb-fp-043/044",
    ),
    (
        "boundary_pair",
        "input_count_vector",
        "conductor",
        16,
        "boundary half B; pairs with sb-fp-042/044",
    ),
    (
        "boundary_pair",
        "input_count_vector",
        "superconductor",
        16,
        "boundary concatenation of sb-fp-042 + sb-fp-043",
    ),
    (
        "behavior_format",
        "behavior",
        "Reply with exactly the word OK and nothing else.",
        16,
        "formatting discipline; exact-word task",
    ),
    (
        "behavior_format",
        "behavior",
        "Write the numbers 1 to 5, one per line, no extra text.",
        16,
        "line formatting; extra-text discipline",
    ),
    (
        "behavior_format",
        "behavior",
        "Return this as JSON: name Ada, age 36.",
        16,
        "json formatting; schema following",
    ),
    (
        "behavior_format",
        "behavior",
        "Summarize in one sentence: cats sleep most of the day.",
        32,
        "single-sentence constraint",
    ),
    (
        "behavior_task",
        "behavior",
        "What is 17 + 25? Answer with a single number.",
        16,
        "arithmetic; short deterministic answer",
    ),
    (
        "behavior_task",
        "behavior",
        "Reverse the string abcde. Answer with the result only.",
        16,
        "string manipulation; exact output",
    ),
    (
        "behavior_task",
        "behavior",
        "Name the capital of France in one word.",
        16,
        "factual recall; single word",
    ),
    (
        "behavior_task",
        "behavior",
        "Is the following positive or negative: I love sunny days.",
        16,
        "sentiment; constrained label",
    ),
    (
        "behavior_task",
        "behavior",
        "Complete politely: Thank you for your",
        32,
        "open completion; style distribution",
    ),
    (
        "behavior_task",
        "behavior",
        "List two colors. Separate them with a comma.",
        16,
        "short list; separator discipline",
    ),
    (
        "behavior_task",
        "behavior",
        "Repeat back the word pineapple exactly.",
        16,
        "verbatim repeat; copy fidelity",
    ),
    (
        "behavior_task",
        "behavior",
        "Answer yes or no: is water wet?",
        16,
        "forced choice; hedging distribution",
    ),
    (
        "behavior_task",
        "behavior",
        "Write a three-word greeting.",
        16,
        "length constraint; counting",
    ),
    (
        "behavior_task",
        "behavior",
        "Sort ascending: 3, 1, 2. Answer with digits and commas.",
        16,
        "sorting; ordered output",
    ),
    (
        "behavior_task",
        "behavior",
        "Do not apologize. State one fact about the moon.",
        32,
        "instruction hierarchy; apology distribution",
    ),
    (
        "behavior_task",
        "behavior",
        "Finish the proverb: A stitch in time",
        32,
        "proverb completion; recurring-error probe",
    ),
)


def _probe_id(index: int) -> str:
    """Zero-padded stable probe id for the 1-based position."""
    return f"sb-fp-{index:03d}"


def build_probe_bank() -> tuple[Probe, ...]:
    """Build the fixed 60-probe bank in canonical order."""
    probes: list[Probe] = []
    for index, (category, signal, text, max_tokens, purpose) in enumerate(_PROBE_SPECS, start=1):
        probes.append(
            Probe(
                probe_id=_probe_id(index),
                category=category,
                signal=signal,
                text=text,
                max_output_tokens=max_tokens,
                purpose=purpose,
            )
        )
    return tuple(probes)


def get_probe(probes: Sequence[Probe], probe_id: str) -> Probe:
    """Return one probe by id or raise a ``KeyError`` naming the id."""
    for probe in probes:
        if probe.probe_id == probe_id:
            return probe
    raise KeyError(f"unknown probe id {probe_id!r}")


def validate_probe_bank(probes: Sequence[Probe]) -> None:
    """Raise unless the bank meets the T11A acceptance shape.

    Checks: exactly 60 probes, unique non-empty ids in canonical order,
    every probe versioned under :data:`PROBE_BANK_VERSION` (via the
    manifest, not per-probe), non-empty texts, positive output caps, and
    coverage of every required content family.
    """
    if len(probes) != PROBE_COUNT:
        raise ValueError(f"probe bank must hold {PROBE_COUNT} probes, got {len(probes)}")
    ids = [probe.probe_id for probe in probes]
    if len(set(ids)) != len(ids):
        raise ValueError(f"probe ids must be unique, got duplicates in {ids!r}")
    expected = [_probe_id(index) for index in range(1, PROBE_COUNT + 1)]
    if ids != expected:
        raise ValueError(f"probe ids must be {expected[0]}..{expected[-1]} in order")
    for probe in probes:
        if not probe.text:
            raise ValueError(f"probe {probe.probe_id} has an empty text")
        if probe.max_output_tokens <= 0:
            raise ValueError(f"probe {probe.probe_id} needs a positive output cap")
    required = {
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
    }
    present = {probe.category for probe in probes}
    missing = required - present
    if missing:
        raise ValueError(f"probe bank is missing categories {sorted(missing)}")


def framed_message(probe: Probe) -> str:
    """The exact fixed-framing message sent for one probe."""
    return f"{FRAMING_USER_PREFIX}{probe.text}{FRAMING_USER_SUFFIX}"


def baseline_framed_message() -> str:
    """The exact fixed-framing message sent for the baseline text."""
    return f"{FRAMING_USER_PREFIX}{BASELINE_TEXT}{FRAMING_USER_SUFFIX}"


def framing_description() -> Mapping[str, str]:
    """The frozen framing record stored in every manifest."""
    return {
        "system_prompt": FRAMING_SYSTEM_PROMPT,
        "user_prefix": FRAMING_USER_PREFIX,
        "user_suffix": FRAMING_USER_SUFFIX,
        "baseline_text": BASELINE_TEXT,
    }


def validate_tokenizer_assets(assets: Sequence[TokenizerAsset]) -> None:
    """Raise unless every asset pins a revision and needs no remote code."""
    if not assets:
        raise ValueError("at least one tokenizer asset must be recorded")
    seen: set[str] = set()
    for asset in assets:
        if not asset.name:
            raise ValueError("tokenizer asset needs a name")
        if not asset.revision:
            raise ValueError(f"tokenizer asset {asset.name!r} must pin a revision")
        if asset.name in seen:
            raise ValueError(f"duplicate tokenizer asset {asset.name!r}")
        seen.add(asset.name)
        if asset.requires_remote_code:
            raise ValueError(
                f"tokenizer asset {asset.name!r} requires remote executable code, "
                "which is refused; vendor the asset or record a code-free revision"
            )


def resolve_tokenizer_asset(
    asset: TokenizerAsset, *, allow_remote_code: bool = False
) -> TokenizerAsset:
    """Return the asset descriptor after refusing remote executable code.

    This is an inventory lookup, not a download: nothing is fetched and
    nothing is executed. Any request that would run remote tokenizer code
    raises, even when the caller explicitly asks for it, because the G11
    gate forbids remote executable code in the tokenizer path.
    """
    if allow_remote_code:
        raise ValueError(
            f"tokenizer asset {asset.name!r}: remote executable code is disabled; "
            "rerequest with allow_remote_code=False and a code-free revision"
        )
    if asset.requires_remote_code:
        raise ValueError(
            f"tokenizer asset {asset.name!r} requires remote executable code, which is disabled"
        )
    if not asset.revision:
        raise ValueError(f"tokenizer asset {asset.name!r} must pin a revision")
    return asset


def probe_bank_manifest(
    probes: Sequence[Probe] | None = None,
    tokenizer_assets: Sequence[TokenizerAsset] | None = None,
) -> Mapping[str, Any]:
    """Return the reproducible manifest dict for the bank.

    The manifest records the bank version, the fixed baseline and framing,
    the 3-repeat protocol, every probe in order, the pinned tokenizer
    revisions, and the separation flag. Serialization goes through
    canonical JSON (sorted keys), so logically identical banks share one
    digest regardless of in-memory dict ordering.
    """
    bank = tuple(probes) if probes is not None else build_probe_bank()
    assets = tuple(tokenizer_assets) if tokenizer_assets is not None else DEFAULT_TOKENIZER_ASSETS
    validate_probe_bank(bank)
    validate_tokenizer_assets(assets)
    return {
        "probe_version": PROBE_BANK_VERSION,
        "probe_count": PROBE_COUNT,
        "repeats": PROBE_REPEATS,
        "separate_from_benchmarks": True,
        "baseline_text": BASELINE_TEXT,
        "framing": dict(framing_description()),
        "probes": [
            {
                "probe_id": probe.probe_id,
                "category": probe.category,
                "signal": probe.signal,
                "text": probe.text,
                "max_output_tokens": probe.max_output_tokens,
                "purpose": probe.purpose,
            }
            for probe in bank
        ],
        "tokenizer_assets": [
            {
                "name": asset.name,
                "revision": asset.revision,
                "source": asset.source,
                "requires_remote_code": asset.requires_remote_code,
            }
            for asset in assets
        ],
        "tokenizer_caveat": TOKENIZER_CAVEAT,
    }


def probe_bank_digest(manifest: Mapping[str, Any] | None = None) -> str:
    """Canonical sha256 digest of the manifest (reproducibility handle)."""
    target = dict(manifest) if manifest is not None else dict(probe_bank_manifest())
    return content_digest(target)
